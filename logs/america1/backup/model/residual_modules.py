# part of the code is borrowed from https://github.com/lllyasviel/ControlNet/blob/ed85cd1e25a5ed592f7d8178495b4483de0331bf/ldm/modules/diffusionmodules/util.py#L177

import torch
import torch.nn as nn
from copy import deepcopy


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class ResidualConnection(nn.Module):
    def __init__(self, main_branch, residual_branch, channel, dim, first_layer_finetune=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_branch = deepcopy(main_branch)
        self.residual_branch = deepcopy(residual_branch)
        
        self.dim = dim
        if dim == 0:
            self.zero_conv = nn.Linear(channel, channel)
        elif dim == 1:
            self.zero_conv = nn.Conv1d(channel, channel, kernel_size=1, padding=0)
        elif dim == 2:
            self.zero_conv = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        elif dim == 3:
            self.zero_conv = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        else:
            raise ValueError
        
        self.zero_conv = zero_module(self.zero_conv)
        
        self.first_layer_finetune = first_layer_finetune
        if first_layer_finetune:
            self.input_adapter = zero_module(nn.Conv2d(32, 32, 1))
        
    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
            
        if self.first_layer_finetune:
            x_ = self.input_adapter(self.residual_branch.conv1(x1))
        
        with torch.no_grad():
            self.main_branch.eval()
            if self.first_layer_finetune:
                x_ = x_ + self.main_branch.conv1(x1)
                # x_ = self.main_branch.conv1(x1)
                def stem(x):
                    x = self.main_branch.relu1(self.main_branch.bn1(x))
                    x = self.main_branch.relu2(self.main_branch.bn2(self.main_branch.conv2(x)))
                    x = self.main_branch.relu3(self.main_branch.bn3(self.main_branch.conv3(x)))
                    x = self.main_branch.avgpool(x)
                    return x
                x1 = stem(x_)
                x1 = self.main_branch.layer1(x1)
                x1 = self.main_branch.layer2(x1)
                x1 = self.main_branch.layer3(x1)
                x1 = self.main_branch.layer4(x1)
                x1 = self.main_branch.attnpool(x1)
            else:
                x1 = self.main_branch(x1)

        if self.first_layer_finetune:
            def stem(x):
                x = self.residual_branch.relu1(self.residual_branch.bn1(x))
                x = self.residual_branch.relu2(self.residual_branch.bn2(self.residual_branch.conv2(x)))
                x = self.residual_branch.relu3(self.residual_branch.bn3(self.residual_branch.conv3(x)))
                x = self.residual_branch.avgpool(x)
                return x
            x2 = stem(x_)
            x2 = self.residual_branch.layer1(x2)
            x2 = self.residual_branch.layer2(x2)
            x2 = self.residual_branch.layer3(x2)
            x2 = self.residual_branch.layer4(x2)
            x2 = self.residual_branch.attnpool(x2)
        else:
            x2 = self.residual_branch(x2)

        x3 = self.zero_conv(x2)

        if len(x1.shape) == 3:
            x3 = x3.unsqueeze(1)
        x4 = x3 + x1
        
        return x4
        
if __name__ == "__main__":
    x = torch.rand(1, 32, 10)
    model1 = nn.Conv1d(32, 2, 3)
    model2 = model1
    model = ResidualConnection(model1, model2, channel=2, dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    print(model(x))
    print(model1(x))
    
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
    opt.step()
    
    print(model(x))
    print(model1(x))    
    
    
     