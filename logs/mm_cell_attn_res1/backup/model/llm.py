'''
Adapted from https://github.com/hellomuffin/exif-as-language/blob/main/model_wrapper.py 
'''
from transformers import DistilBertModel, DistilBertConfig
from transformers import AlbertModel, AlbertConfig
from transformers import RobertaModel,RobertaConfig
import clip
import torch.nn as nn
import torch

def bert(pretrained=True):
    if pretrained:
        return DistilBertModel.from_pretrained("distilbert-base-uncased")
    else:
        return DistilBertModel(DistilBertConfig())
    
def albert(pretrained=True):
    if pretrained:
        return AlbertModel.from_pretrained("albert-base-v2")
    else:
        return AlbertModel(AlbertConfig)

def roberta(pretrained=True):
    if pretrained:
        return RobertaModel.from_pretrained("roberta-base")
    else:
        return RobertaModel(RobertaConfig)

class ClipTransformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model, _ = clip.load("RN50", device='cpu')
        self.transformer = model
    
    def forward(self, x):
        if len(x.shape) == 3:
            batch_size, cell_number, _ = x.shape
            x = x.flatten(0, 1)
            x = self.transformer(x)
            if cell_number!= 1:
                x = x.view(batch_size, cell_number, -1)
        else:
            x = self.transformer(x)
        return x
            

def clip_bert():
    return ClipTransformer()

if __name__ == "__main__":
    model = clip_bert()
    print(model)