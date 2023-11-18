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
    def __init__(self, hidden_dim, nhead=1):
        super(AttentivePooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear_mapping_img = nn.Linear(hidden_dim, 128, bias=False)
        self.linear_mapping_tab = nn.Linear(hidden_dim, 128, bias=False)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.fc = nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, 1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                    )
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, x):
        # x.shape = batch_size, seq_len, hidden_dim
        # corr_weights means the correspondence between imaging and tabular data
        # attention_weights means the importance of the fused imaging and tabular data
        image_emb = x[:,0,...]
        tab_emb = x[:,1:,...]
        image_proj = self.linear_mapping_img(image_emb.unsqueeze(1)).transpose(1,2)
        tab_proj = self.linear_mapping_tab(tab_emb)
        corr = torch.bmm(tab_proj, image_proj)
        corr_weights = torch.softmax(corr, dim=1)*tab_emb.size(1)
        tab_emb = tab_emb + image_emb.unsqueeze(1)*corr_weights # fusion of imaging and tabular features
        
        x = self.encoder_layer(tab_emb) + tab_emb + image_emb.unsqueeze(1)
        attention_weights = self.attention(x)
        attention_weights = torch.sigmoid(attention_weights)
        
        tab_emb = self.fc(tab_emb.transpose(1,2)).transpose(1,2)
        output = torch.bmm(attention_weights.transpose(1, 2), tab_emb).squeeze(1)
        return output

class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
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