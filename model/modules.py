import torch.nn as nn
import torch

class Conv1x1(nn.Module):
    """
    1x1 convolutional layer
    args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        bn (bool): whether use batch normalization
        act_fn (nn.Module): activation function
    """
    def __init__(self, in_channels, out_channels, bn=False, act_fn=nn.Identity()) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        self.act_fn = act_fn
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_fn(x)
        return x
    
class Conv3x3(nn.Module):
    """
    3x3 convolutional layer
    args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        bn (bool): whether use batch normalization
        act_fn (nn.Module): activation function
    """
    def __init__(self, in_channels, out_channels, bn=False, act_fn=nn.Identity()) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        self.act_fn = act_fn
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_fn(x)
        return x

class DenseLayer(nn.Module):
    """
    fully-connected layer
    args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        bn (bool): whether use batch normalization
        act_fn (nn.Module): activation function
    """
    def __init__(self, in_channels, out_channels, bn=False, act_fn=nn.Identity()) -> None:
        super().__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = nn.Identity()
        self.act_fn = act_fn

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_fn(x)
        return x

if __name__ == "__main__":
    x = torch.rand(3, 100)
    mlp = MLP(100, 3, [])
    print(mlp(x).shape)