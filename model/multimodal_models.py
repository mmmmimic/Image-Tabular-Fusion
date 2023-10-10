import torch.nn as nn
from .resnet import resnet50
from .mlps import MLP
import torch

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
    pass

class MultimodelCell(nn.Module):
    pass
    
if __name__ == '__main__':
    import torch    
    image = torch.rand(16, 3, 224, 224)
    tab_line = torch.rand(16, 63)
    model = MultimodalBaseline(tab_emb_dim=63, num_classes=286, emb_dim=2048, fusion='add')
    logit = model(image=image, tab_line=tab_line)
    print(logit.shape)
    print(logit)
    