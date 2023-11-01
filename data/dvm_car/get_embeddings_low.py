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


# add embedder into path
sys.path.append('/home/lmx/Image-Tabular-Fusion/data')
from tabular_utils import OneHotEmbedder, TextEmbedder, Scarf, RandomMask
from dataset import DVMLow

def get_all(dataset, batch_size):
    lines = []
    loader = DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size)
    for item in tqdm(loader):
        tab_line = item['tab_line']
        lines.append(tab_line.numpy())
    lines = np.concatenate(lines, axis=0)
    return lines

def clip_get_line(dataset, batch_size):
    lines = []
    loader = DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size)
    model, _ = clip.load('RN50', device='cuda')
    for item in tqdm(loader):
        tab_line = item['tab_line']
        b = tab_line.size(0)
        col_num = tab_line.size(1)
        tab_line = tab_line.flatten(0, 1)
        with torch.no_grad():
            line = model.encode_text(tab_line.long().cuda())
            line = line.view((b, col_num, -1))
            lines.append(line.cpu().numpy())
    lines = np.concatenate(lines, axis=0)
    return lines

def clip_get_tab(dataset, batch_size):
    lines = []
    loader = DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size)
    model, _ = clip.load('RN50', device='cuda')
    for item in tqdm(loader):
        tab_line = item['tab_line']
        tab_line = tab_line.squeeze(1)
        with torch.no_grad():
            line = model.encode_text(tab_line.long().cuda())
            lines.append(line.cpu().numpy())
    lines = np.concatenate(lines, axis=0)
    return lines  

def onehot_noaug01():
    # onehot, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVMLow(split='train', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'], ratio=0.1)
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_noaug01_train.npy', embs)

def onehot_noaug001():
    # onehot, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVMLow(split='train', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'], ratio=0.01)
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_noaug001_train.npy', embs)

def text_cell01():
    # text, cellwise, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = DVMLow(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'], ratio=0.1)
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_cell01_train.npy', embs)

def text_cell001():
    # text, cellwise, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = DVMLow(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'], ratio=0.01)
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_cell001_train.npy', embs)

def text_cell_context01():
    # text, cellwise, with context
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = DVMLow(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', context='car'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'], ratio=0.1)
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_context01_train.npy', embs)

def text_cell_context001():
    # text, cellwise, with context
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = DVMLow(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', context='car'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'], ratio=0.01)
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_context001_train.npy', embs)

def text_gpt01():
    # text, gpt template, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = DVMLow(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip', chatgpt_tmpl='/home/lmx/Image-Tabular-Fusion/data/dvm_car/chatgpt_tmpl.txt'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'], ratio=0.1)
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_gpt01_train.npy', embs)

def text_gpt001():
    # text, gpt template, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = DVMLow(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip', chatgpt_tmpl='/home/lmx/Image-Tabular-Fusion/data/dvm_car/chatgpt_tmpl.txt'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'], ratio=0.01)
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_gpt001_train.npy', embs)

def text_tab_clip01():
    # text, tabwise, clip, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVMLow(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'], ratio=0.1)
    embs = clip_get_tab(trainset, batch_size=32)
    np.save('clip_tab01_train.npy', embs)

def text_tab_clip001():
    # text, tabwise, clip, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVMLow(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'], ratio=0.01)
    embs = clip_get_tab(trainset, batch_size=32)
    np.save('clip_tab001_train.npy', embs)
            
if __name__ == "__main__":
    onehot_noaug01()
    onehot_noaug001()
    text_cell01()
    text_cell001()
    text_cell_context01()
    text_cell_context001()
    text_gpt01()
    text_gpt001()
    text_tab_clip01()
    text_tab_clip001()
    