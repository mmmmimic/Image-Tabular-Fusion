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

class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentivePooling, self).__init__()
        self.hidden_dim = hidden_dim
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        # x.shape = batch_size, seq_len, hidden_dim
        # x = self.encoder_layer(x) # feature interaction between different modalities
        attention_weights = self.attention(x)  # batch_size, seq_len, 1
        attention_weights = torch.softmax(attention_weights, dim=1)
        output = attention_weights * x  # batch_size, seq_len, hidden_dim
        output = output.sum(dim=1)  # batch_size, hidden_dim
        return output

class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.attention = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # x.shape = batch_size, seq_len, input_dim
        x = self.encoder_layer(x)
        attention_weights = self.attention(x)  # batch_size, seq_len, hidden_dim
        attention_weights = torch.softmax(attention_weights, dim=1)

        gate_weights = self.gate(x)  # batch_size, seq_len, hidden_dim
        gate_weights = torch.sigmoid(gate_weights)

        output = attention_weights * gate_weights * x  # batch_size, seq_len, input_dim
        output = output.sum(dim=1)  # batch_size, input_dim
        return output


if __name__ == "__main__":
    x = torch.rand(3, 100, 512)
    pool = AttentivePooling(100, 512)
    # pool = GatedAttention(512, 512)
    print(pool(x).shape, x)