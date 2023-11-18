'''
Adapted from https://github.com/hellomuffin/exif-as-language/blob/main/model_wrapper.py 
'''
from transformers import DistilBertModel, DistilBertConfig
from transformers import AlbertModel, AlbertConfig
from transformers import RobertaModel,RobertaConfig
import clip

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
    
def clip_():
    model, _ = clip.load("RN50", device='cpu')
    return model.transformer

if __name__ == "__main__":
    model = clip_()
    print(model)