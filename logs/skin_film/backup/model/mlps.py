import torch
from .modules import DenseLayer, Conv1x1
import torch.nn as nn
import clip

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
        
    def forward(self, x):
        x = self.fc(x)
        return x

class ClipMLP(nn.Module):
    """
    multi-layer perceptron encoding with CLIP (for ablation study)
    args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        hid_channels (list): number of hidden channels
        bn (bool): whether use batch normalization
        act_fn (nn.Module): activation function
    """
    def __init__(self, out_channels, hid_channels, bn=False, act_fn=nn.ReLU()) -> None:
        super().__init__()
        
        # self.clip_encoder, _ = clip.load("RN50", device='cpu')
        self.mlp = MLP(1024*17, out_channels, hid_channels, bn, act_fn)
        
    def forward(self, x):
        # x [batch, num_cell, 77]
        # batch_size = x.size(0)
        # multi_cell = x.size(1) > 1
        
        # if multi_cell:
        #     num_cell = x.size(1)
        #     x = x.flatten(0, 1)
        # else:
        #     x = x.squeeze(1)
            
        # with torch.no_grad():
        #     x = self.clip_encoder.encode_text(x.long()) # [batch, num_cell, 1024]
            
        # if multi_cell:
        #     x = torch.reshape(x, (batch_size, num_cell, 1024))
        #     # x = torch.mean(x, dim=1)
        #     x = x.flatten(-2)
        x = x.flatten(-2)
        x = self.mlp(x)
        return x

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
    
    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == "__main__":
    pass