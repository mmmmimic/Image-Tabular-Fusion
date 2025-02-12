import torch.nn as nn
import torch
from copy import deepcopy
from .residual_modules import zero_module
import math

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

class DinoFusion(nn.Module):
    def __init__(self, feat_dim, phi=0.5):
        super(DinoFusion, self).__init__()
        self.feat_dim = feat_dim
        self.phi = phi
        reduction = 2

        self.fc1 = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim//reduction),
                    nn.GELU(),
                    nn.Linear(feat_dim//reduction, feat_dim, bias=False)
                    )
        self.fc1[2] = zero_module(self.fc1[2])
        self.fc2 =nn.Sequential(
                    nn.Linear(feat_dim, feat_dim//reduction),
                    nn.GELU(),
                    nn.Linear(feat_dim//reduction, feat_dim, bias=False)
                    )
        self.fc2[2] = zero_module(self.fc2[2])
        
        self.fc1_teacher = deepcopy(self.fc1)
        self.fc2_teacher = deepcopy(self.fc2)
        for param in self.fc1_teacher.parameters():
            param.requires_grad = False
        for param in self.fc2_teacher.parameters():
            param.requires_grad = False

        self.gcn1 = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim//reduction),
                    nn.GELU(),
                    nn.Linear(feat_dim//reduction, feat_dim),
                    nn.GELU()
        )
        ## text-text
        self.gcn2 = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim//reduction),
                    nn.GELU(),
                    nn.Linear(feat_dim//reduction, feat_dim),
                    nn.GELU()
        )
        ## image-text
        self.gcn3 = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim//reduction),
                    nn.GELU(),
                    nn.Linear(feat_dim//reduction, feat_dim),
                    nn.GELU()
        )
        ## text-image
        self.gcn4 = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim//reduction),
                    nn.GELU(),
                    nn.Linear(feat_dim//reduction, feat_dim),
                    nn.GELU()
        )

        # self.fc3 = nn.Sequential(
        #             nn.Dropout(0.3),
        #             nn.Linear(feat_dim, feat_dim),
        #             nn.GELU(),
        #             nn.Linear(feat_dim, feat_dim)
        # )

        self.logit_scale = torch.tensor(4.6052)

        # self.pool = nn.AdaptiveAvgPool1d(16)
        self.fc3 = nn.TransformerEncoderLayer(d_model=feat_dim*2, nhead=4, batch_first=True)
        self.fc4 = nn.TransformerEncoderLayer(d_model=feat_dim*2, nhead=4, batch_first=True)
        self.fc5 = nn.Linear(feat_dim*2, feat_dim)

    def forward(self, image_tokens, tab_tokens):
        ## normalize
        # image_tokens = torch.cat((image_tokens[:,[0],:], self.pool(image_tokens[:,1:,:].transpose(1,2)).transpose(1,2)), dim=1) # downsample image patches to a certain number
        logit_scale = self.logit_scale.exp()
        image_tokens = image_tokens + self.fc1(image_tokens)*self.phi + self.fc1_teacher(image_tokens)*(1-self.phi)
        image_tokens_ = image_tokens / image_tokens.norm(dim=1, keepdim=True)

        tab_tokens = tab_tokens + self.fc2(tab_tokens)*self.phi + self.fc2_teacher(tab_tokens)*(1-self.phi)
        tab_tokens_ = tab_tokens / tab_tokens.norm(dim=1, keepdim=True)

        self.fc1_teacher[0].weight = deepcopy(self.fc1[0].weight)
        self.fc1_teacher[2].weight = deepcopy(self.fc1[2].weight)
        self.fc2_teacher[0].weight = deepcopy(self.fc2[0].weight)
        self.fc2_teacher[2].weight = deepcopy(self.fc2[2].weight)


        ## to use gnn, we need a graph
        ## affinity matrix
        # the first epoch, construct the affinity matrix
        aff_mtx1 = torch.bmm(image_tokens_[:,1:,:], image_tokens_[:,1:,:].transpose(1,2))*logit_scale # B, N-1, N-1
        aff_mtx1 = (aff_mtx1 + aff_mtx1.transpose(1,2))/2 # symmetric
        # aff_mtx1 = aff_mtx1 / aff_mtx1.max()
        aff_mtx1 = torch.softmax(aff_mtx1, dim=-1)

        aff_mtx2 = torch.bmm(tab_tokens_, tab_tokens_.transpose(1,2))*logit_scale # B, M, M
        aff_mtx2 = (aff_mtx2 + aff_mtx2.transpose(1,2))/2
        # 创建对角线掩码矩阵
        diag_mask = torch.eye(aff_mtx2.shape[-1], dtype=torch.bool).unsqueeze(0).expand(aff_mtx2.shape[0], -1, -1).to(aff_mtx2.device)
        # 使用掩码将对角线置零
        aff_mtx2[diag_mask] = -1e12
        # aff_mtx2 = aff_mtx2 / aff_mtx2.max()
        aff_mtx2 = torch.softmax(aff_mtx2, dim=-1)

        aff_mtx3 = torch.bmm(image_tokens_, tab_tokens_.transpose(1,2))*logit_scale # B, N, M
        # aff_mtx3 = aff_mtx3 / aff_mtx3.max()
        aff_mtx3 = torch.softmax(aff_mtx3, dim=-1)

        aff_mtx4 = torch.bmm(tab_tokens_, image_tokens_[:,1:,:].transpose(1,2))*logit_scale # B, M, N
        # aff_mtx4 = aff_mtx4 / aff_mtx4.max()
        aff_mtx4 = torch.softmax(aff_mtx4, dim=-1)

        ## gcns
        emb1 = self.gcn1(image_tokens[:,1:,:])
        emb1 = torch.bmm(aff_mtx1, image_tokens[:,1:,:]) + image_tokens[:,1:,:]
        emb2 = self.gcn2(tab_tokens)
        emb2 = torch.bmm(aff_mtx2, emb2) + tab_tokens
        # embs1 = torch.cat((self.gcn1(image_tokens[:,[0],:]) + image_tokens[:,[0],:], emb1, emb2), dim=1)

        ## interaction
        node1 = self.gcn3(tab_tokens)
        node1 = torch.bmm(aff_mtx3, node1) + image_tokens
        node1 = (node1[:,[0],:] + node1[:,1:,:])/2 # local + global

        node2 = self.gcn4(image_tokens[:,1:,:])
        node2 = torch.bmm(aff_mtx4, node2) + tab_tokens
        # embs2 = torch.cat((node2, node1), dim=1)

        # embs =  torch.cat((embs1, embs2), dim=-1)
        embs1 = torch.cat((node1, emb1), dim=-1)
        embs1 = self.fc3(embs1)
        embs1 = torch.mean(embs1, dim=1)
        embs2 = torch.cat((node2, emb2), dim=-1)
        embs2 = self.fc4(embs2)
        embs2 = torch.mean(embs2, dim=1)
        embs = embs1*torch.sigmoid(embs2) + torch.sigmoid(embs1)*embs2        
        ans = self.fc5(embs)
        return ans

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
