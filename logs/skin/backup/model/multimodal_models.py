import torch.nn as nn
from .resnet import resnet50
from .mlps import MLP
import torch
from .residual_modules import ResidualConnection
from .modules import AttentivePooling, DinoFusion
from .llm import clip_bert
from .resnet import clip_resnet50_encoder, medclip_resnet50_encoder , pubmedclip_resnet50_encoder
from .vit import clip_vit_b_16_encoder, clip_vit_b_32_encoder
from .mmdynamics import MMDynamic
from peft import LoraConfig, get_peft_model

class MultimodalModel(nn.Module):
    def __init__(self, tab_emb_dim, num_classes, feat_dim = 1024, fusion='cat', 
                 image_encoder='rn50', tab_encoder='mlp', frozen_tab=True, 
                 complex_classifier = False, nhead=1,
                 first_layer_finetune=False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        '''
        Structure:
            image -> image_encoder -> image features   |
                                                        | -> fusion -> classifier
            table -> tabular_encoder -> table features  |
        '''
        self.tab_emb_dim = tab_emb_dim
        self.feat_dim = feat_dim # joint feature dimension
        self.first_layer_finetune = first_layer_finetune
        self.encoder_name = image_encoder
        self.frozen_tab = frozen_tab
        
        self._build_image_encoder(image_encoder)
        self._build_tabular_encoder(tab_encoder)

        self.fusion = fusion
        self.nhead = nhead
        self._init_fuser()
        
        if complex_classifier:
            self.classifier = nn.Sequential(
                            nn.Linear(self.fused_dim, self.fused_dim),
                            nn.BatchNorm1d(self.fused_dim),
                            nn.LeakyReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(self.fused_dim, self.fused_dim),
                            nn.BatchNorm1d(self.fused_dim),
                            nn.LeakyReLU(),
                            nn.Linear(self.fused_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(self.fused_dim, num_classes)
    

    def _init_fuser(self):
        if self.fusion == 'add':
            self.fused_dim = self.feat_dim
            self.fuse = lambda x, y: x + y
        
        elif self.fusion == 'cat':
            self.fused_dim = self.feat_dim * 2
            self.fuse = lambda x, y: torch.cat((x.flatten(1), y.flatten(1)), dim=-1)
            
        elif self.fusion == 'mul':
            self.fused_dim = self.feat_dim
            self.fuse = lambda x, y: x * y
        
        elif self.fusion == 'attn':
            self.fused_dim = self.feat_dim
            self.attentive_pool = AttentivePooling(self.feat_dim, nhead=self.nhead)
            self.fuse = lambda x, y: self.attentive_pool(torch.cat((x.view(x.size(0), -1, x.size(-1)), 
                                                                    y.view(y.size(0), -1, y.size(-1))), dim=1))      
        elif self.fusion == 'dino':
            self.fused_dim = self.feat_dim
            self.dino_fusion = DinoFusion(self.feat_dim)
            self.fuse = lambda x, y: self.dino_fusion(x, y)
        
        else:
            raise NotImplementedError      

    def _build_image_encoder(self, encoder_name):
        if encoder_name == 'rn50':
            model = resnet50(pretrained=False, num_classes=self.feat_dim)
        elif encoder_name == 'rn50_clip':
            model = clip_resnet50_encoder()
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'
        elif encoder_name == 'vitb16_clip':
            model = clip_vit_b_16_encoder()
            assert self.feat_dim == 512,f'embedding dimension {self.feat_dim} is illegal'  
            lora_config = LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["attn"],
                lora_dropout=0.5,
            )

            model = get_peft_model(model, lora_config)    
        elif encoder_name == 'vitb32_clip':
            model = clip_vit_b_32_encoder()
            assert self.feat_dim == 512,f'embedding dimension {self.feat_dim} is illegal'
            lora_config = LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["attn"],
                lora_dropout=0.5,
            )

            model = get_peft_model(model, lora_config)         
        else:
            raise ValueError   
        
        self.image_encoder = model      
        
    def _build_tabular_encoder(self, encoder_name):
        if encoder_name == 'mlp':
            model = MLP(self.tab_emb_dim, self.feat_dim, [self.feat_dim])
        elif encoder_name == 'identical':
            model = nn.Identity()
        elif encoder_name == 'clip':
            model = clip_bert()
            assert self.feat_dim == 512,f'embedding dimension {self.feat_dim} is illegal'
            if self.frozen_tab:
                for param in model.parameters():
                    param.requires_grad = False
                model.prefix_embedding.requires_grad = True
            # lora_config = LoraConfig(
            #     r=4,
            #     lora_alpha=16,
            #     target_modules=["attn"],
            #     lora_dropout=0.5,
            # )

            # model = get_peft_model(model, lora_config)
        elif encoder_name == 'clip_res':
            model_ = clip_bert()
            model = ResidualConnection(model_, model_, channel=1024, dim=2) # deepcopy is integrated inside the module
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'
        elif encoder_name == 'clip_mlp_res':
            model_ = clip_bert()
            mlp_model = MLP(self.tab_emb_dim, self.feat_dim, [self.feat_dim])
            model = ResidualConnection(model_, mlp_model, channel=1024, dim=0) # deepcopy is integrated inside the module
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'
        elif encoder_name == 'preload_res':
            model_ = clip_bert()
            model = ResidualConnection(nn.Identity(), model_, channel=1024, dim=0) # deepcopy is integrated inside the module
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'    
        elif encoder_name == 'preload_mlp_res':
            model_ = nn.Identity()
            mlp_model = MLP(self.tab_emb_dim, self.feat_dim, [self.feat_dim])
            model = ResidualConnection(model_, mlp_model, channel=1024, dim=0) # deepcopy is integrated inside the module
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'                     
        else:
            raise ValueError

        self.tab_encoder = model

    def encode_image(self, x):
        if 'clip' in self.encoder_name:
            if 'rn' not in self.encoder_name:
                x = self.image_encoder.conv1(x)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0,2,1)
                x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
                x = x + self.image_encoder.positional_embedding.to(x.dtype)
                x = self.image_encoder.ln_pre(x)

                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.image_encoder.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.image_encoder.ln_post(x)
        
                if self.image_encoder.proj is not None:
                    x = x @ self.image_encoder.proj
            else:
                def stem(x):
                    x = self.image_encoder.relu1(self.image_encoder.bn1(self.image_encoder.conv1(x)))
                    x = self.image_encoder.relu2(self.image_encoder.bn2(self.image_encoder.conv2(x)))
                    x = self.image_encoder.relu3(self.image_encoder.bn3(self.image_encoder.conv3(x)))
                    x = self.image_encoder.avgpool(x)
                    return x
                x = stem(x)
                x = self.image_encoder.layer1(x)
                x = self.image_encoder.layer2(x)
                x = self.image_encoder.layer3(x)
                x = self.image_encoder.layer4(x)
                x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
                x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
                x = x + self.image_encoder.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
                x, _ = nn.functional.multi_head_attention_forward(
                query=x[0], key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.image_encoder.attnpool.num_heads,
                q_proj_weight=self.image_encoder.attnpool.q_proj.weight,
                k_proj_weight=self.image_encoder.attnpool.k_proj.weight,
                v_proj_weight=self.image_encoder.attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.image_encoder.attnpool.q_proj.bias, self.image_encoder.attnpool.k_proj.bias, self.image_encoder.attnpool.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.image_encoder.attnpool.c_proj.weight,
                out_proj_bias=self.image_encoder.attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.image_encoder.attnpool.training,
                need_weights=False
            )
                x = x.permute(1,0,2)
        else:
            x = self.image_encoder(x)
        return x          
    
    def encode_tabline(self, tab_line):
        x = self.tab_encoder(tab_line)
        return x
    
    def forward(self, tab_line, image, *args, **kwargs):
        tab_feat = self.encode_tabline(tab_line)
        image_feat = self.encode_image(image)
        
        fused_feat = self.fuse(image_feat, tab_feat)

        logit = self.classifier(fused_feat)
        
        return logit

class LogitEmbeddingLayer(nn.Module):
    def __init__(self, feat_dim = 1024, zero_conv = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.embedder = nn.Sequential(
            nn.Conv1d(32, 128, 1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, feat_dim, 1)
            )
        
        if zero_conv:
            self.zero_conv = nn.Linear(feat_dim, feat_dim)
            for p in self.zero_conv.parameters():
                p.detach().zero_()
        else:
            self.zero_conv = nn.Identity()
    
    def forward(self, line_tab):
        if len(line_tab.shape) == 2:
            line_tab = line_tab.unsqueeze(-1)
        emb = self.embedder(line_tab.transpose(1,2)).transpose(1,2) # b, n, feat_dim
        emb = self.zero_conv(emb)
        
        return emb

class ResMultimodalModel(MultimodalModel):
    def __init__(self, tab_emb_dim, num_classes, feat_dim = 1024, fusion='cat', 
                 image_encoder='rn50', tab_encoder='mlp', frozen_tab=True, 
                 complex_classifier = False, nhead=1, zero_conv = True,
                 source_idx = None, 
                 target_idx = None,
                 first_layer_finetune = True,
                 *args, **kwargs) -> None:
        super().__init__(tab_emb_dim, num_classes, feat_dim, fusion, 
                 image_encoder, tab_encoder, frozen_tab, 
                 complex_classifier, nhead, first_layer_finetune, *args, **kwargs)
        self.embedder = LogitEmbeddingLayer(feat_dim=feat_dim, zero_conv=zero_conv)
        self.source_idx = source_idx
        self.target_idx = target_idx
        
    def forward(self, main_tabline, res_tabline, image, *args, **kwargs):
        image_feat = self.encode_image(image)
        tab_feat = self.encode_tabline(main_tabline)
        
        # res_tabline: b, n
        if isinstance(self.target_idx, list):
            source_idx = torch.tensor(self.source_idx)
            target_idx = torch.tensor(self.target_idx)
            res_tabline = res_tabline[:,source_idx,...]
            res_tab_feat = self.embedder(res_tabline)
            for i, t in enumerate(target_idx):
                tab_feat[:, t, ...] = tab_feat[:, t, ...] + res_tab_feat[:,i,...]
        else:
            res_tab_feat = self.embedder(res_tabline)
            tab_feat = tab_feat + res_tab_feat
        
        fused_feat = self.fuse(image_feat, tab_feat)
        
        logit = self.classifier(fused_feat)
        
        return logit        

class IRene(nn.Module):
    def __init__(self, tab_num, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(nhead=12, d_model=768, 
                                                                                          batch_first=True,
                                                                                          dropout=0.3
                                                                                          ), 
                                                 num_layers=2,
                                                 )
        self.linear_mapping = nn.ModuleList()
        for i in range(tab_num):
            self.linear_mapping.append(nn.Linear(1, 768))
            
        self.dp = nn.Dropout(0.2)

        self.img_patchize = nn.Conv2d(3, 768, 16, 16)
        self.fc = nn.Linear(768, num_classes)
        
    def forward(self, image, tab_line, *args, **kwargs):
        image = self.img_patchize(image).flatten(-2).permute(0,2,1) # [B, 768]
        tab_emb = []
        tab_line = tab_line.unsqueeze(-1)
        for i in range(tab_line.size(1)):
            tab_emb.append(self.dp(self.linear_mapping[i](tab_line[:,i,...]) + tab_line[:,i,...]).unsqueeze(1))
        emb = torch.cat(tab_emb, dim=1)
        emb = torch.cat((emb, image), dim=1)
        feat = self.transformer(emb)
        
        feat = torch.mean(feat, dim=1)
        logit = self.fc(feat)
        
        return logit    
  
class TriModalModel(MultimodalModel):
    def __init__(self, tab_emb_dim, num_classes, feat_dim=1024, fusion='cat', image_encoder='rn50', tab_encoder='mlp', *args, **kwargs) -> None:
        super().__init__(tab_emb_dim, num_classes, feat_dim, fusion, image_encoder, tab_encoder, frozen_tab=False, *args, **kwargs)
        self.classifier = nn.Linear(3*1024, num_classes)
        
    def encode_tabline(self, embd):
        embd = torch.nan_to_num(embd, nan=0.0) # if nan, replace it
        return super().encode_tabline(embd)
        
    def forward(self, image, text, embd, *args, **kwargs):
        # tab_line should include [clip_embeddings, onehot_embeddings, continous_embeddings]
        tab_feat = self.encode_tabline(embd)
        image_feat = self.encode_image(image)
        
        fused_feat = torch.cat((tab_feat, image_feat, text), dim=-1)
        
        logit = self.classifier(fused_feat)
        
        return logit   

class MMDynamicModel(MultimodalModel):
    def __init__(self, tab_emb_dim, num_classes, feat_dim = 1024, 
                 fusion='cat', image_encoder='rn50', tab_encoder='mlp', 
                 frozen_tab=True, *args, **kwargs) -> None:
        super().__init__(tab_emb_dim, num_classes, feat_dim = 1024, 
                 fusion='cat', image_encoder='rn50', tab_encoder='mlp', 
                 frozen_tab=True, *args, **kwargs)
        self.classifier = MMDynamic([feat_dim, feat_dim], [feat_dim], num_classes, 0.2)

    def forward(self, tab_line, image, label, *args, **kwargs):
        tab_feat = self.encode_tabline(tab_line)
        image_feat = self.encode_image(image)

        fused_feat = [tab_feat, image_feat]
        
        loss, logit = self.classifier(fused_feat, label)
        
        return {'loss': loss, 'logit': logit}    
    
class PPNet(nn.Module):
    def __init__(self, tab_emb_dim, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(), 
                    nn.Conv2d(64, 64, 3, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, 3, stride=2, padding=1)
        )
        self.layer4 = nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, 3, stride=2, padding=1)
        )
        
        self.fc1 = nn.Linear(tab_emb_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 256)
        
        self.classifier = nn.Sequential(
                    nn.Linear(12544, 512),
                    nn.ReLU(),
                    nn.Linear(512, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
        )
        
    @staticmethod
    def _multiply(x, y):
        return x.unsqueeze(-1).unsqueeze(-1)*y
        
    def forward(self, tab_line, image, *args, **kwargs):
        feat = self.conv1(image)
        tab_feat = self.fc1(tab_line)
        feat = self._multiply(tab_feat, feat)
        
        feat = self.layer1(feat)
        tab_feat = self.fc2(tab_feat)
        feat = self._multiply(tab_feat, feat)

        feat = self.layer2(feat)
        tab_feat = self.fc3(tab_feat)
        feat = self._multiply(tab_feat, feat)

        feat = self.layer3(feat)
        tab_feat = self.fc4(tab_feat)
        feat = self._multiply(tab_feat, feat)        

        feat = self.layer4(feat)
        tab_feat = self.fc5(tab_feat)
        feat = self._multiply(tab_feat, feat)
        
        feat = feat.flatten(1)
        logit = self.classifier(feat)
        
        return logit
        
if __name__ == '__main__':
    import torch    
    image = torch.rand(16, 3, 224, 224).cuda()
    tab_line = torch.rand(16, 17).cuda()
    model = IRene(tab_num=17, num_classes=286).cuda()
    print(model(image, tab_line).shape)
    