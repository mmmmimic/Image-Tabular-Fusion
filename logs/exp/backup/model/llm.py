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
        model, _ = clip.load("ViT-B/32", device='cpu')
        self.transformer = model
        del self.transformer.visual
        # 假设 CLIP 文本编码器的嵌入维度为 512
        embedding_dim = 512
        prefix_length = 16  # 可调节
        self.prefix_embedding = nn.Parameter(torch.randn(1, prefix_length, embedding_dim))

    @staticmethod
    def add_prefix_to_input(prefix_embedding, text_embeddings):
        """
        prefix_embedding: [1, prefix_length, embedding_dim]
        text_embeddings: [batch_size, seq_length, embedding_dim]
        """
        batch_size = text_embeddings.size(0)
        # 扩展 prefix_embedding 以匹配 batch_size
        prefix = prefix_embedding.expand(batch_size, -1, -1)
        # 拼接 prefix 和原始文本嵌入
        return torch.cat([prefix[:,:prefix.size(1)//2,:], text_embeddings[:,:-prefix.size(1),:],prefix[:,prefix.size(1)//2:,:]], dim=1)

    def encode_text(self, text):
        x = self.transformer.token_embedding(text)  # [batch_size, n_ctx, d_model]
        ## add trainable predix token
        x = self.add_prefix_to_input(self.prefix_embedding, x)
        x = x + self.transformer.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.transformer.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.transformer.text_projection

        return x

    def forward(self, x):
        if len(x.shape) == 3:
            batch_size, cell_number, _ = x.shape
            x = x.flatten(0, 1)
            x = self.encode_text(x)
            if cell_number!= 1:
                x = x.view(batch_size, cell_number, -1)
        else:
            x = self.encode_text(x)
        return x
            

def clip_bert():
    return ClipTransformer()

if __name__ == "__main__":
    model = clip_bert()
    print(model)