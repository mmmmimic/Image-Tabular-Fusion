import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

class tabtransformer(nn.Module):
    def __init__(self, categories, num_continuous, dim, dim_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        avg, std = torch.zeros((num_continuous, 1)), torch.ones((num_continuous, 1))
        avgstd = torch.cat((avg, std), dim=-1)
        self.model = TabTransformer(
        categories=categories,
        num_continuous= num_continuous,
        dim=dim,
        dim_out=dim_out,
        depth=6,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_mults= (4, 2),
        mlp_act=nn.ReLU(),
        continuous_mean_std=avgstd,
        *args, **kwargs
    )
        
    def forward(self, tab_line, *args, **kwargs):
        for k in tab_line.keys():
            tab_line[k] = tab_line[k].cuda()
        return self.model(**tab_line)
        

if __name__ == "__main__":
    model = TabTransformer(
        categories=(10, 5, 6, 5, 8),
        num_continuous= 10,
        dim=32,
        dim_out=286,
        depth=6,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_mults= (4, 2),
        mlp_act=nn.ReLU(),
        continuous_mean_std=torch.randn(10, 2)
    )
    x_cat = torch.randint(0, 5, (1, 5))
    x_cont = torch.randn(1, 10)
    pred = model(x_cat, x_cont)
    
    print(model)
    