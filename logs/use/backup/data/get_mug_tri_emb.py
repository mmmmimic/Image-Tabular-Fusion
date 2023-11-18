import numpy as np
import torch
import clip
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from os.path import join

# add embedder into path
# sys.path.append('/home/lmx/Image-Tabular-Fusion/data')
from tabular_utils import OneHotEmbedder, TextEmbedder, Scarf, RandomMask
from dataset import MUGTri

ROOT = '/uac/rshr/mxlin/src/mxlin/Image-Tabular-Fusion/data/mug_dataset/'

def clip_get_tab(dataset, batch_size):
    texts = []
    onehots = []
    loader = DataLoader(dataset, shuffle=False, num_workers=6, batch_size=batch_size)
    model, _ = clip.load('RN50', device='cuda')
    for item in tqdm(loader):
        text, embd = item['text'], item['embd']
        tab_line = clip.tokenize(text, truncate=True)
        with torch.no_grad():
            line = model.encode_text(tab_line.long().cuda())
            texts.append(line.cpu().numpy())
            onehots.append(embd.numpy())
    texts = np.concatenate(texts, axis=0)
    onehots = np.concatenate(onehots, axis=0)
    return texts, onehots

def go(token='Pokemon-primary_type'):
    root_path = join(ROOT, token)
    # text, tabwise, clip, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = MUGTri(split='train', transforms=transforms, 
                      root_dir=root_path, modal=['tab'])
    text, onehot = clip_get_tab(trainset, batch_size=32)
    np.save(join(root_path, 'text_train.npy'), text)
    np.save(join(root_path, 'onehot_train.npy'), onehot)
    

    devset = MUGTri(split='val', transforms=transforms, 
                      root_dir=root_path, modal=['tab'])
    text, onehot = clip_get_tab(devset, batch_size=32)
    np.save(join(root_path, 'text_val.npy'), text)
    np.save(join(root_path, 'onehot_val.npy'), onehot)

    testset = MUGTri(split='test', transforms=transforms, 
                      root_dir=root_path, modal=['tab'])
    text, onehot = clip_get_tab(testset, batch_size=32)
    np.save(join(root_path, 'text_test.npy'), text)
    np.save(join(root_path, 'onehot_test.npy'), onehot)
    
if __name__ == "__main__":
    go(token='Pokemon-primary_type')
    go(token='Pokemon-secondary_type')
    go(token='LeagueOfLegends-Skin-category')
    go(token='CSGO-Skin-quality')
    go(token='Hearthstone-All-cardClass')
    go(token='Hearthstone-All-set')
    go(token='Hearthstone-Minion-race')
    go(token='Hearthstone-Spell-spellSchool')
    