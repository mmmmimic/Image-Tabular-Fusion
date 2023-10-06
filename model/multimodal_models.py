import torch.nn as nn
from resnet import resnet50

class MultimodalBaseline(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, data):
        pass
    
if __name__ == '__main__':
    model = MultimodalBaseline()
    