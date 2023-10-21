import torch.nn as nn
from .resnet import resnet50
from .mlps import MLP
import torch
import clip
from .residual_modules import ResidualConnection
from .modules import AttentivePooling
from .llm import clip_bert
from .resnet import clip_resnet50_encoder  

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
            model = ResidualConnection(model_, model_, channel=1024, dim=0) # deepcopy is integrated inside the module
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
    
    
if __name__ == '__main__':
    import torch    
    image = torch.rand(16, 3, 224, 224)
    tab_line = torch.rand(16, 63)
    model = None
    