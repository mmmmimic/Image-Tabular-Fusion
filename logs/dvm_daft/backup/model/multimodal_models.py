import torch.nn as nn
from .resnet import resnet50
from .mlps import MLP
import torch
import clip
from .residual_modules import ResidualConnection
from .modules import AttentivePooling
from .llm import clip_bert
from .resnet import clip_resnet50_encoder, medclip_resnet50_encoder 
from .vit import clip_vit_b_32_encoder

class MultimodalModel(nn.Module):
    def __init__(self, tab_emb_dim, num_classes, feat_dim = 1024, fusion='cat', image_encoder='rn50', tab_encoder='mlp', frozen_tab=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        '''
        Structure:
            image -> image_encoder -> image features   |
                                                        | -> fusion -> classifier
            table -> tabular_encoder -> table features  |
        '''
        self.tab_emb_dim = tab_emb_dim
        self.feat_dim = feat_dim # joint feature dimension
        self._build_image_encoder(image_encoder)
        self._build_tabular_encoder(tab_encoder)

        self.fusion = fusion
        self._init_fuser()
        
        self.classifier = nn.Linear(self.fused_dim, num_classes)
        
        self.frozen_tab = frozen_tab

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
            self.attentive_pool = AttentivePooling(self.feat_dim)
            self.fuse = lambda x, y: self.attentive_pool(torch.cat((x.view(x.size(0), -1, x.size(-1)), y.view(y.size(0), -1, y.size(-1))), dim=1))
        
        else:
            raise NotImplementedError      

    def _build_image_encoder(self, encoder_name):
        if encoder_name == 'rn50':
            model = resnet50(pretrained=False, num_classes=self.feat_dim)
        elif encoder_name == 'rn50_clip':
            model = clip_resnet50_encoder()
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'
        elif encoder_name == 'rn50_clip_res':
            model_ = clip_resnet50_encoder()
            model = ResidualConnection(model_, model_, channel=self.feat_dim, dim=0) # deepcopy is integrated inside the module
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'
        elif encoder_name == 'rn50_medclip_res':
            model_ = medclip_resnet50_encoder()
            model = ResidualConnection(model_, model_, channel=self.feat_dim, dim=0) # deepcopy is integrated inside the module
            assert self.feat_dim == 512,f'embedding dimension {self.feat_dim} is illegal'
        elif encoder_name == 'vitb32_clip_res':
            model = clip_vit_b_32_encoder()
            model = ResidualConnection(model_, model_, channel=self.feat_dim, dim=0) # deepcopy is integrated inside the module
            assert self.feat_dim == 768,f'embedding dimension {self.feat_dim} is illegal'            
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
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'
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

    def encode_image(self, image):
        x = self.image_encoder(image)
        return x          
    
    def encode_tabline(self, tab_line):
        if self.frozen_tab:
            with torch.no_grad():
                self.tab_encoder.eval()
                x = self.tab_encoder(tab_line)
        else:
            x = self.tab_encoder(tab_line)
        return x
    
    def forward(self, tab_line, image, *args, **kwargs):
        tab_feat = self.encode_tabline(tab_line)
        image_feat = self.encode_image(image)
        
        fused_feat = self.fuse(image_feat, tab_feat)
        
        logit = self.classifier(fused_feat)
        
        return logit

class ResMultimodalModel(MultimodalModel):
    def __init__(self, tab_emb_dim, num_classes, feat_dim=1024, fusion='cat', image_encoder='rn50', tab_encoder='mlp', *args, **kwargs) -> None:
        super().__init__(tab_emb_dim, num_classes, feat_dim, fusion, image_encoder, tab_encoder, *args, **kwargs)
        
    def _build_tabular_encoder(self, encoder_name):
        if encoder_name == 'mlp':
            model = MLP(self.tab_emb_dim, self.feat_dim, [self.feat_dim])
        elif encoder_name == 'clip_bert':
            model = clip_bert()                  
        else:
            raise ValueError
        
        self.tab_encoder = ResidualConnection(nn.Identity(), model, channel=self.feat_dim, dim=0)      
        # self.tab_encoder = ResidualConnection(clip_bert(), model)      
    
    def encode_tabline(self, main_tabline, res_tabline):
        x = self.tab_encoder(main_tabline, res_tabline)
        return x
    
    def forward(self, main_tabline, res_tabline, image, *args, **kwargs):
        # tab_line should include [clip_embeddings, onehot_embeddings, continous_embeddings]
        tab_feat = self.encode_tabline(main_tabline, res_tabline[:,:13])
        image_feat = self.encode_image(image)
        
        fused_feat = self.fuse(image_feat, tab_feat)
        
        logit = self.classifier(fused_feat)
        
        return logit        
    
class DAFT(nn.Module):
    def __init__(self, tab_emb_dim, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        resnet = clip_resnet50_encoder()
        self.image_encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu1,
            resnet.conv2,
            resnet.bn2,
            resnet.relu2,
            resnet.conv3,
            resnet.bn3,
            resnet.relu3,
            resnet.avgpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.attn_pool = resnet.attnpool
        
        self.daft = nn.Sequential(
          nn.Linear(tab_emb_dim + 2048, (tab_emb_dim + 2048)//7),
          nn.ReLU(),
          nn.Linear((tab_emb_dim + 2048)//7, 4096)  
        )
        
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, tab_line, image, *args, **kwargs):
        feat_before_daft = self.image_encoder(image)
        feat_maps = self.pool(feat_before_daft).flatten(2).squeeze(-1)
        fused_feat = torch.cat((feat_maps, tab_line), dim=1)
        daft_scores = self.daft(fused_feat)
        alpha = daft_scores[:,:2048]
        beta = daft_scores[:,2048:]
        feat_after_daft = alpha.unsqueeze(-1).unsqueeze(-1) * feat_before_daft + beta.unsqueeze(-1).unsqueeze(-1)
        feat = self.attn_pool(feat_after_daft)
        logit = self.classifier(feat)
        return logit

class Film(nn.Module):
    def __init__(self, tab_emb_dim, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        resnet = clip_resnet50_encoder()
        self.image_encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu1,
            resnet.conv2,
            resnet.bn2,
            resnet.relu2,
            resnet.conv3,
            resnet.bn3,
            resnet.relu3,
            resnet.avgpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.attn_pool = resnet.attnpool
        
        self.film = nn.Sequential(
          nn.Linear(tab_emb_dim, tab_emb_dim //7),
          nn.ReLU(),
          nn.Linear(tab_emb_dim//7, 4096)  
        )
        
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, tab_line, image, *args, **kwargs):
        feat_before_daft = self.image_encoder(image)
        daft_scores = self.film(tab_line)
        alpha = daft_scores[:,:2048]
        beta = daft_scores[:,2048:]
        feat_after_daft = alpha.unsqueeze(-1).unsqueeze(-1) * feat_before_daft + beta.unsqueeze(-1).unsqueeze(-1)
        feat = self.attn_pool(feat_after_daft)
        logit = self.classifier(feat)
        return logit

class PPNet(nn.Module):
    def __init__(self, tab_emb_dim, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        resnet = clip_resnet50_encoder()
        
        self.pool = resnet.attnpool
        
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu1,
            resnet.conv2,
            resnet.bn2,
            resnet.relu2,
            resnet.conv3,
            resnet.bn3,
            resnet.relu3,
            resnet.avgpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.fc1 = nn.Linear(tab_emb_dim, resnet.conv3.out_channels)
        self.fc2 = nn.Linear(resnet.conv3.out_channels, resnet.layer1[-1].conv3.out_channels)
        self.fc3 = nn.Linear(resnet.layer1[-1].conv3.out_channels, resnet.layer2[-1].conv3.out_channels)
        self.fc4 = nn.Linear(resnet.layer2[-1].conv3.out_channels, resnet.layer3[-1].conv3.out_channels)
        self.fc5 = nn.Linear(resnet.layer3[-1].conv3.out_channels, resnet.layer4[-1].conv3.out_channels)
        
        self.classifier = nn.Linear(1024, num_classes)
        
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
        
        feat = self.pool(feat)
        logit = self.classifier(feat)
        
        return logit
        
if __name__ == '__main__':
    import torch    
    image = torch.rand(16, 3, 224, 224)
    tab_line = torch.rand(16, 63)
    model = None
    