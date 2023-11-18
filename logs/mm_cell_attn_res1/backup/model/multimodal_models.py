import torch.nn as nn
from .resnet import resnet50
from .mlps import MLP
import torch
import clip
from .residual_modules import ResidualConnection
from .modules import AttentivePooling
from .llm import clip_bert
from .resnet import clip_resnet50_encoder

class MultimodalBaseline(nn.Module):
    def __init__(self, tab_emb_dim, num_classes, emb_dim = 1024, fusion='add', image_encoder='rn50', tab_encoder='mlp', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if image_encoder == 'rn50':
            self.image_encoder = resnet50(pretrained=False, num_classes=emb_dim)
        else:
            raise ValueError

        if tab_encoder == 'mlp':
            self.tab_encoder = MLP(tab_emb_dim, emb_dim, [emb_dim])
        else:
            raise ValueError

        self.fusion = fusion
        self.emb_dim = emb_dim
        self._init_fusion()
        
        self.head = nn.Linear(self.fused_dim, num_classes)
        
    def forward(self, tab_line, image, *args, **kwargs):
        tab_emb = self.tab_encoder(tab_line)
        image_emb = self.image_encoder(image)
        
        fused_emb = self.fuse(image_emb, tab_emb)
        
        logit = self.head(fused_emb)
        
        return logit
    
    def _init_fusion(self):
        if self.fusion == 'add':
            self.fused_dim = self.emb_dim
            self.fuse = lambda x, y: x + y
        
        elif self.fusion == 'cat':
            self.fused_dim = self.emb_dim * 2
            self.fuse = lambda x, y: torch.cat((x, y), dim=-1)
            
        elif self.fusion == 'mul':
            self.fused_dim = self.emb_dim
            self.fuse = lambda x, y: x * y
        
        else:
            raise NotImplementedError        
            
class MultimodalSentence(nn.Module):
    def __init__(self, num_classes, emb_dim = 1024, fusion='add', freeze_encoder=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        clip_model, _ = clip.load('RN50', device='cpu')
        self.image_encoder = clip_model

        self.fusion = fusion
        self.emb_dim = emb_dim
        self._init_fusion()
        
        self.head = MLP(self.fused_dim, num_classes, [emb_dim])
        self.freeze_encoder = freeze_encoder
        # nn.Linear(self.fused_dim, num_classes)
        
    def forward(self, tab_line, image, *args, **kwargs):
        tab_emb = tab_line
        if self.freeze_encoder:
            self.image_encoder.eval()
            with torch.no_grad():
                image_emb = self.image_encoder.encode_image(image)
        else:
            image_emb = self.image_encoder.encode_image(image)
        fused_emb = self.fuse(image_emb, tab_emb)
        
        logit = self.head(fused_emb)
        
        return logit
    
    def _init_fusion(self):
        if self.fusion == 'add':
            self.fused_dim = self.emb_dim
            self.fuse = lambda x, y: x + y
        
        elif self.fusion == 'cat':
            self.fused_dim = self.emb_dim * 2 # 1536
            self.fuse = lambda x, y: torch.cat((x, y), dim=-1)
            
        elif self.fusion == 'mul':
            self.fused_dim = self.emb_dim
            self.fuse = lambda x, y: x * y
        
        else:
            raise NotImplementedError        

class MultimodelCell(nn.Module):
    def __init__(self, num_classes, emb_dim = 1024, fusion='add', freeze_encoder=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        clip_model, _ = clip.load('RN50', device='cpu')
        self.image_encoder = clip_model

        self.fusion = fusion
        self.emb_dim = emb_dim
        self._init_fusion()
        
        self.head = MLP(self.fused_dim, num_classes, [emb_dim])
        self.freeze_encoder = freeze_encoder
        # nn.Linear(self.fused_dim, num_classes)
        
    def forward(self, tab_line, image, *args, **kwargs):
        tab_emb = tab_line.flatten(1)
        if self.freeze_encoder:
            self.image_encoder.eval()
            with torch.no_grad():
                image_emb = self.image_encoder.encode_image(image)
        else:
            image_emb = self.image_encoder.encode_image(image)
            
        fused_emb = self.fuse(image_emb, tab_emb)
        
        logit = self.head(fused_emb)
        
        return logit
    
    def _init_fusion(self):
        if self.fusion == 'add':
            self.fused_dim = self.emb_dim
            self.fuse = lambda x, y: x + y
        
        elif self.fusion == 'cat':
            self.fused_dim = self.emb_dim * 18 # 1536
            self.fuse = lambda x, y: torch.cat((x, y), dim=-1)
            
        elif self.fusion == 'mul':
            self.fused_dim = self.emb_dim
            self.fuse = lambda x, y: x * y
        
        else:
            raise NotImplementedError        

class MultimodalModel(nn.Module):
    def __init__(self, tab_emb_dim, num_classes, feat_dim = 1024, fusion='cat', image_encoder='rn50', tab_encoder='mlp', *args, **kwargs) -> None:
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
            self.fuse = lambda x, y: self.attentive_pool(torch.cat((x.unsqueeze(1), y), dim=1))
        
        else:
            raise NotImplementedError      

    def _build_image_encoder(self, encoder_name):
        if encoder_name == 'rn50':
            model = resnet50(pretrained=False, num_classes=self.feat_dim)
        elif encoder_name == 'rn50_clip':
            model = clip_resnet50_encoder()
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'
        elif encoder_name == 'rn_50_clip_res':
            model_ = clip_resnet50_encoder()
            model = ResidualConnection(model_, model_, channel=1024, dim=2) # deepcopy is integrated inside the module
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'
        else:
            raise ValueError        
        
        self.image_encoder = model      
        
    def _build_tabular_encoder(self, encoder_name):
        if encoder_name == 'mlp':
            model = MLP(self.tab_emb_dim, self.feat_dim, [self.feat_dim])
        elif encoder_name == 'identical':
            model = nn.Identity()
        elif encoder_name == 'clip':
            model, _ = clip_bert()
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'
        elif encoder_name == 'clip_res':
            model_, _ = clip_bert()
            model = ResidualConnection(model_, model_, channel=1024, dim=2) # deepcopy is integrated inside the module
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'
        elif encoder_name == 'clip_mlp_res':
            model_, _ = clip_bert()
            mlp_model = MLP(self.tab_emb_dim, self.feat_dim, [self.feat_dim])
            model = ResidualConnection(model_, mlp_model, channel=1024, dim=0) # deepcopy is integrated inside the module
            assert self.feat_dim == 1024,f'embedding dimension {self.feat_dim} is illegal'
        elif encoder_name == 'preload_res':
            model_, _ = clip_bert()
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
        with torch.no_grad():
            self.tab_encoder.eval()
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
    
    
if __name__ == '__main__':
    import torch    
    image = torch.rand(16, 3, 224, 224)
    tab_line = torch.rand(16, 63)
    model = MultimodalBaseline(tab_emb_dim=63, num_classes=286, emb_dim=2048, fusion='add')
    logit = model(image=image, tab_line=tab_line)
    print(logit.shape)
    print(logit)
    