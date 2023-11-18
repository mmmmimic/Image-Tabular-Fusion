import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from data import DVM, OneHotEmbedder, DVMPre
import numpy as np
import pandas as pd
import torchvision.transforms as T
from model import MLP
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torchvision


batch_size = 32

tab_transform = lambda x, y: y

transforms = {
    'tab_tf': tab_transform, 
    'img_tf': T.RandomChoice(
        [
            T.Compose(
                [
                   T.RandomApply([
                    T.ColorJitter(brightness=[0.2, 1.8], contrast=[0.2, 1.8], saturation=[0.2, 1.8], hue=0)], p=0.8),
                    T.RandomGrayscale(p=0.2),
                    T.RandomApply([T.GaussianBlur(kernel_size=(29, 29), sigma=(0.1, 2.0))], p=0.5),
                    T.RandomResizedCrop(size=(128, 128), scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor()
                ]
            ),
            T.Compose(
        [
            T.Resize((128, 128)),
            T.ToTensor()
        ]
        )
        ], p=[0.95, 0.05]
        )
}

# trainset = DVM(split='train', transforms=transforms, numerical=True, tab_embedder=OneHotEmbedder(), modal=['img'])
trainset = DVMPre(split='train', transforms=transforms, kwd='onehot_noaug', modal=['tab'])
trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=False)

transforms = {
    'tab_tf': tab_transform, 
    'img_tf': T.Compose(
        [
            T.Resize((128, 128)),
            T.ToTensor()
        ]
    )
}
# valset = DVM(split='val', transforms=transforms, numerical=True, tab_embedder=OneHotEmbedder(), modal=['img'])
valset = DVMPre(split='val', transforms=transforms, kwd='onehot_noaug', modal=['tab'])

valloader = DataLoader(valset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
# testset = DVM(split='test', transforms=transforms, numerical=True, tab_embedder=OneHotEmbedder(), modal=['img'])
testset = DVMPre(split='test', transforms=transforms, kwd='onehot_noaug', modal=['tab'])

testloader = DataLoader(testset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = MLP(1024*17, 286, [1024], bn=False, act_fn=nn.ReLU())

model = MLP(63, 286, [2048, 2048], bn=False, act_fn=nn.ReLU())
# model = nn.Sequential(
#         nn.Linear(63, 2048),
#         nn.BatchNorm1d(2048),
#         nn.ReLU(),
#         nn.Linear(2048, 2048),
#         nn.ReLU(),
#         nn.Linear(2048, 286)
# )

# model = torchvision.models.resnet.resnet50(pretrained=False)
# model.fc = nn.Linear(model.fc.in_features, 286)

model = model.to(device)


criterion = nn.CrossEntropyLoss()

lr = 1e-4
num_epochs = 500
weight_decay = 0 # 1e-5

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

if __name__ == "__main__":
    # train and val
    best_acc = 0.
    best_epoch = int(0)
    for epoch in range(num_epochs):
        print(f'epoch: {epoch}')
        preds = []
        gts = []
        total_loss = 0.
        total_counts = int(0)
        model.train()
        for batch in tqdm(trainloader):
            data, label = batch['tab_line'], batch['label']
            # data = data.flatten(1,2)
            # data, label = batch['image'], batch['label']
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            
            logit = model(data)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            
            pred = torch.argmax(logit, dim=-1)
            preds.append(pred.detach().cpu().numpy())
            gts.append(label.cpu().numpy())
            total_loss += loss.item()*logit.size(0)
            total_counts += logit.size(0)
        
        scheduler.step()
        preds = np.concatenate(preds, axis=-1)
        gts = np.concatenate(gts, axis=-1)
        total_loss /= total_counts
        
        acc = accuracy_score(gts, preds)
        
        print(f'train------> loss: {total_loss: .2f}, acc: {acc: .4f}')
        
        preds = []
        gts = []
        total_loss = 0.
        total_counts = int(0)
        model.eval()
        for batch in tqdm(valloader):
            data, label = batch['tab_line'], batch['label']
            # data = data.flatten(1,2)
            # data, label = batch['image'], batch['label']
            data, label = data.to(device), label.to(device)
            
            with torch.no_grad():
                logit = model(data)
            loss = criterion(logit, label)
            
            pred = torch.argmax(logit, dim=-1)
            preds.append(pred.detach().cpu().numpy())
            gts.append(label.cpu().numpy())
            total_loss += loss.item()*logit.size(0)
            total_counts += logit.size(0)
        preds = np.concatenate(preds, axis=-1)
        gts = np.concatenate(gts, axis=-1)
        total_loss /= total_counts
        
        acc = accuracy_score(gts, preds)
        
        print(f'val------> loss: {total_loss: .2f}, acc: {acc: .4f}')            
        
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            
        print(f'best acc: {best_acc: .4f}, best epoch: {best_epoch}')
        
    
        
        