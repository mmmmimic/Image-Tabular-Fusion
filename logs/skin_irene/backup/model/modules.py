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

class DinoFusion(nn.Module):
    def __init__(self, feat_dim, phi=0.01, reduction=8, dropout=0.5, ratio=1, latent_dim=64):
        super(DinoFusion, self).__init__()
        assert latent_dim >= reduction,f'The latent dimension {latent_dim} should be larger than the reduction rate {reduction}. '

        self.phi = phi
        self.ratio = ratio

        self.fc1 = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim//reduction),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(feat_dim//reduction, feat_dim, bias=False),
                    nn.GELU()
                    )
        self.fc1[-2] = zero_module(self.fc1[-2])
        self.fc2 =nn.Sequential(
                    nn.Linear(feat_dim, feat_dim//reduction),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(feat_dim//reduction, feat_dim, False),
                    nn.GELU()
                    )
        self.fc2[-2] = zero_module(self.fc2[-2])

        self.latent_map = nn.Sequential(
                    nn.Linear(feat_dim, latent_dim),
                    nn.GELU()
        )
        
        self.fc1_teacher = deepcopy(self.fc1)
        for param in self.fc1_teacher.parameters():
            param.requires_grad = False
        self.fc2_teacher = deepcopy(self.fc2)
        for param in self.fc2_teacher.parameters():
            param.requires_grad = False

        ## image-image
        self.gcn1 = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim//reduction),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim//reduction, latent_dim),
                    nn.GELU()
        )
        ## text-text
        self.gcn2 = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim//reduction),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim//reduction, latent_dim),
                    nn.GELU()
        )
        ## image-text
        self.gcn3 = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim//reduction),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim//reduction, latent_dim),
                    nn.GELU()
        )
        ## text-image
        self.gcn4 = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim//reduction),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim//reduction, latent_dim),
                    nn.GELU()
        )

        self.logit_scale = torch.tensor(4.6052)

        self.attn_pool = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim//reduction),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim//reduction, 1),
                    nn.Sigmoid()
        )

        self.act = nn.GELU()

    def forward(self, image_tokens, tab_tokens):
        ## normalize
        logit_scale = self.logit_scale.exp()
        image_tokens = image_tokens + (self.fc1(image_tokens)*self.phi + self.fc1_teacher(image_tokens)*(1-self.phi))*self.ratio
        image_tokens_ = image_tokens / image_tokens.norm(dim=1, keepdim=True)

        tab_tokens = tab_tokens + (self.fc2(tab_tokens)*self.phi + self.fc2_teacher(tab_tokens)*(1-self.phi))*self.ratio
        tab_tokens_ = tab_tokens / tab_tokens.norm(dim=1, keepdim=True)

        self.fc1_teacher[0].weight = deepcopy(self.fc1[0].weight)
        self.fc1_teacher[3].weight = deepcopy(self.fc1[3].weight)
        self.fc2_teacher[0].weight = deepcopy(self.fc2[0].weight)
        self.fc2_teacher[3].weight = deepcopy(self.fc2[3].weight)

        ## to use gnn, we need a graph
        ## affinity matrix
        # the first epoch, construct the affinity matrix
        aff_mtx1 = torch.bmm(image_tokens_[:,1:,:], image_tokens_[:,1:,:].transpose(1,2))*logit_scale # B, N-1, N-1
        aff_mtx1 = (aff_mtx1 + aff_mtx1.transpose(1,2))/2 # symmetric
        diag_mask = torch.eye(aff_mtx1.shape[-1], dtype=torch.bool).unsqueeze(0).expand(aff_mtx1.shape[0], -1, -1).to(aff_mtx1.device)
        aff_mtx1[diag_mask] = -1e12
        # aff_mtx1 = aff_mtx1 / aff_mtx1.max()
        aff_mtx1 = torch.softmax(aff_mtx1, dim=-1)

        aff_mtx2 = torch.bmm(tab_tokens_, tab_tokens_.transpose(1,2))*logit_scale # B, M, M
        aff_mtx2 = (aff_mtx2 + aff_mtx2.transpose(1,2))/2
        diag_mask = torch.eye(aff_mtx2.shape[-1], dtype=torch.bool).unsqueeze(0).expand(aff_mtx2.shape[0], -1, -1).to(aff_mtx2.device)
        aff_mtx2[diag_mask] = -1e12
        # aff_mtx2 = aff_mtx2 / aff_mtx2.max()
        aff_mtx2 = torch.softmax(aff_mtx2, dim=-1)

        aff_mtx3 = torch.bmm(image_tokens_, tab_tokens_.transpose(1,2))*logit_scale # B, N, M
        # aff_mtx3 = aff_mtx3 / aff_mtx3.max()
        aff_mtx3 = torch.softmax(aff_mtx3, dim=-1)

        aff_mtx4 = torch.bmm(tab_tokens_, image_tokens_[:,1:,:].transpose(1,2))*logit_scale # B, M, N
        # aff_mtx4 = aff_mtx4 / aff_mtx4.max()
        aff_mtx4 = torch.softmax(aff_mtx4, dim=-1)

        image_tokens = self.latent_map(image_tokens)
        tab_tokens = self.latent_map(tab_tokens)

        ## gcns
        image_emb = self.gcn1(image_tokens[:,1:,:])
        image_emb = self.act(torch.bmm(aff_mtx1, image_emb) + image_tokens[:,1:,:])
        tab_emb = self.gcn2(tab_tokens)
        tab_emb = self.act(torch.bmm(aff_mtx2, tab_emb) + tab_tokens)

        ## interaction
        image_tab = self.gcn3(tab_tokens)
        image_tab = self.act(torch.bmm(aff_mtx3, image_tab) + image_tokens)
        image_tab = (image_tab[:,[0],:] + image_tab[:,1:,:])/2 # local + global

        tab_image = self.gcn4(image_tokens[:,1:,:])
        tab_image = self.act(torch.bmm(aff_mtx4, tab_image) + tab_tokens)

        image_emb = torch.sum(self.attn_pool(image_emb)*image_emb, dim=1)
        tab_emb = torch.sum(self.attn_pool(tab_emb)*tab_emb, dim=1)
        image_tab = torch.sum(self.attn_pool(image_tab)*image_tab, dim=1)
        tab_image = torch.sum(self.attn_pool(tab_image)*tab_image, dim=1)

        emb = torch.cat((image_emb, tab_emb, image_tab, tab_image), dim=-1)
        return self.act(emb)

if __name__ == "__main__":
    x = torch.rand(3, 100, 512)
    fusion = DinoFusion(512)
    print(fusion(x).shape, x)