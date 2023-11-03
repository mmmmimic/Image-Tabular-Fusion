from model import ResNetTabAttention
import torch

model = ResNetTabAttention((224, 224),
                           n_frames=1,
                           n_tab=6)
x = torch.randn(8, 3, 128, 128)
tab = torch.randn(8, 6)
model = ResNetTabAttention(input_size=(x.shape[-2], x.shape[-1]), 
                           n_frames=1, n_tab=tab.shape[-1],
                           num_classes=10
                           )
print(model)
print(model(x, tab).shape)