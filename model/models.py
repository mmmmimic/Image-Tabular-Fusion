import torch
from .modules import DenseLayer, Conv1x1, Conv3x3
import torch.nn as nn
from torchvision.models import ResNet

class MLP(nn.Module):
    """
    multi-layer perceptron
    args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        hid_channels (list): number of hidden channels
        bn (bool): whether use batch normalization
        act_fn (nn.Module): activation function
    """
    def __init__(self, in_channels, out_channels, hid_channels, bn=False, act_fn=nn.ReLU()) -> None:
        super().__init__()
        self.fc = nn.ModuleList()
        
        if len(hid_channels):
            for i, hid in enumerate(hid_channels):
                if i == 0:
                    self.fc.append(DenseLayer(in_channels, hid, bn, act_fn))
                else:
                    self.fc.append(DenseLayer(hid_channels[i-1], hid, bn, act_fn))
            self.fc.append(nn.Linear(hid_channels[-1], out_channels))
        else:
            self.fc.append(nn.Linear(in_channels, out_channels))
        
        self.fc = nn.Sequential(*self.fc)
        
    # def forward(self, x, *args, **kwargs):
    #     x = self.fc(x)
    #     return x
    def forward(self, tab_line, *args, **kwargs):
        x = self.fc(tab_line)
        return {'logit': x, 'label': kwargs['label']}

class MLP2D(nn.Module):
    """
    multi-layer perceptron for 2D inputs with shape (B,C,H,W)
    args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        hid_channels (list): number of hidden channels
        bn (bool): whether use batch normalization
        act_fn (nn.Module): activation function
    """
    def __init__(self, in_channels, out_channels, hid_channels, bn=False, act_fn=nn.ReLU()) -> None:
        super().__init__()
        self.fc = nn.ModuleList()
        
        if len(hid_channels):
            for i, hid in enumerate(hid_channels):
                if i == 0:
                    self.fc.append(Conv1x1(in_channels, hid, bn, act_fn))
                else:
                    self.fc.append(Conv1x1(hid_channels[i-1], hid, bn, act_fn))
            self.fc.append(Conv1x1(hid_channels[-1], out_channels, bn, act_fn))
        else:
            self.fc.append(Conv1x1(in_channels, out_channels, bn, act_fn))
        
        self.fc = nn.Sequential(*self.fc)
    
    def forward(self, x, *args, **kwargs):
        x = self.fc(x)
        return x

if __name__ == "__main__":
    pass