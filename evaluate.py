import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pandas as pd
from data import DVM, OneHotEmbedder, DefaultEmbedder
from matplotlib import pyplot as plt
from torchvision import transforms as T
from collections import Counter
import torchvision
from torch.utils.data import DataLoader


# class DenseLayer(nn.Module):
#     """
#     fully-connected layer
#     args:
#         in_channels (int): number of input channels
#         out_channels (int): number of output channels
#         bn (bool): whether use batch normalization
#         act_fn (nn.Module): activation function
#     """
#     def __init__(self, in_channels, out_channels, bn=False, act_fn=nn.Identity()) -> None:
#         super().__init__()
#         self.conv = nn.Linear(in_channels, out_channels)
#         if bn:
#             self.bn = nn.BatchNorm1d(out_channels)
#         else:
#             self.bn = nn.Identity()
#         self.act_fn = act_fn

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act_fn(x)
#         return x
    
# class MLP(nn.Module):
#     """
#     multi-layer perceptron
#     args:
#         in_channels (int): number of input channels
#         out_channels (int): number of output channels
#         hid_channels (list): number of hidden channels
#         bn (bool): whether use batch normalization
#         act_fn (nn.Module): activation function
#     """
#     def __init__(self, in_channels, out_channels, hid_channels, bn=False, act_fn=nn.ReLU()) -> None:
#         super().__init__()
#         self.fc = nn.ModuleList()
        
#         if len(hid_channels):
#             for i, hid in enumerate(hid_channels):
#                 if i == 0:
#                     self.fc.append(DenseLayer(in_channels, hid, bn, act_fn))
#                 else:
#                     self.fc.append(DenseLayer(hid_channels[i-1], hid, bn, act_fn))
#             self.fc.append(nn.Linear(hid_channels[-1], out_channels))
#         else:
#             self.fc.append(nn.Linear(in_channels, out_channels))
        
#         self.fc = nn.Sequential(*self.fc)
        
#     def forward(self, x):
#         x = self.fc(x)
#         return x
    
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = MLP(63, 286, [2048, 2048], bn=False)
    
#     def forward(self, x):
#         return self.model(x)
       
# model = Model()

# class Model(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.encoder  = nn.Sequential(
#                         nn.Linear(63, 2048),
#                         nn.BatchNorm1d(2048),
#                         nn.ReLU(),
#                         nn.Linear(2048, 2048)
#                     )
#         self.classifier = nn.Linear(2048, 286)
        
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.classifier(x)
#         return x

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = torchvision.models.resnet.resnet50()
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(2048, 286)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
model = Model()
# model.load_state_dict(torch.load('/home/lmx/Image-Tabular-Fusion/logs/mlp_baseline/models/best_model_1697208346_7856207_epoch103.t7')['state_dict'])
# state_dict = torch.load('/home/lmx/MMCL-Tabular-Imaging/runs/eval/mlp/checkpoint_best_acc.ckpt')['state_dict']
state_dict = torch.load('/home/lmx/MMCL-Tabular-Imaging/runs/eval/resnet2/checkpoint_best_acc.ckpt')['state_dict']
for k in list(state_dict.keys()):
    if 'classifier' not in k:
        # k_ = k.replace('model.encoder.encoder', 'encoder')
        k_ = k.replace('model.encoder', 'encoder')
    else:
        k_ = k.replace('model.', '')
    state_dict[k_] = state_dict[k]
    del state_dict[k]
    
# model.load_state_dict(torch.load('/home/lmx/MMCL-Tabular-Imaging/runs/eval/polar-leaf-35/checkpoint_best_acc.ckpt')['state_dict'])
model.load_state_dict(state_dict)

tab_transform = lambda x, y: y
    
transforms = {
    'tab_tf': tab_transform, 
    'img_tf': T.Compose(
        [
            T.Resize((128, 128)),
            T.ToTensor()
        ]
    )
}

# data = DVM(split='train', transforms=transforms, numerical=True, tab_embedder=OneHotEmbedder(), modal=['tab'])
data = DVM(split='val', transforms=transforms, numerical=True, tab_embedder=OneHotEmbedder(), modal=['img'])
loader = DataLoader(data, batch_size=64, num_workers=8, shuffle=False)

model = model.cuda()
preds = []
labels = []
# label_ = torch.load('/home/lmx/MMCL-Tabular-Imaging/data/dvm/19586296/features/labels_model_all_train_all_views.pt')
# data_ = torch.load('/home/lmx/MMCL-Tabular-Imaging/data.pt')

# for i in tqdm(range(len(data))):
for x in tqdm(loader):
    # x = data[i]
    # d, l = x['tab_line'], x['label']
    d, l = x['image'], x['label']
    # d, l = data_[i], label_[i] 
    model.eval()
    with torch.no_grad():
        logits = model(d.cuda())
    pred = torch.argmax(logits, dim=-1)
    preds.append(pred.cpu().numpy())
    labels.append(l.numpy())

preds = np.concatenate(preds, axis=0)
labels = np.concatenate(labels, axis=0)
# preds = np.array(preds)
# labels = np.array(labels)
print(accuracy_score(labels, preds), balanced_accuracy_score(labels, preds))


# 模型相同
# label 相同， label没问题，是preds不同
# debug来看，pred, logit, tabline也相同
# -3 is different
# acc, avg_acc不同