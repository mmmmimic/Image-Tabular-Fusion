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
from dataset import MUG

ROOT = '/home/lmx/Image-Tabular-Fusion/data/mug_dataset/Pokemon-secondary_type'

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

def text_cell():
    # text, cellwise, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = MUG(split='train', transforms=transforms,
                tab_embedder=TextEmbedder(cellwise=True, model='clip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_cell_train.npy', embs)

    trainset = MUG(split='val', transforms=transforms,
                tab_embedder=TextEmbedder(cellwise=True, model='clip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_cell_val.npy', embs)

    trainset = MUG(split='test', transforms=transforms,
                tab_embedder=TextEmbedder(cellwise=True, model='clip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_cell_test.npy', embs)

def text_cell_context():
    # text, cellwise, with context
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = MUG(split='train', transforms=transforms,
                tab_embedder=TextEmbedder(cellwise=True, model='clip', context='pokemon'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_context_train.npy', embs)

    trainset = MUG(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', context='pokemon'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_context_val.npy', embs)

    trainset = MUG(split='test', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', context='pokemon'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_context_test.npy', embs)

def text_gpt():
    # text, gpt template, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = MUG(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip', chatgpt_tmpl='/home/lmx/Image-Tabular-Fusion/data/dvm_car/chatgpt_tmpl.txt'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_gpt_train.npy', embs)

    trainset = MUG(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', chatgpt_tmpl='/home/lmx/Image-Tabular-Fusion/data/dvm_car/chatgpt_tmpl.txt'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_gpt_val.npy', embs)

    trainset = MUG(split='test', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', chatgpt_tmpl='/home/lmx/Image-Tabular-Fusion/data/dvm_car/chatgpt_tmpl.txt'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_gpt_test.npy', embs)

def text_tab_clip():
    # text, tabwise, clip, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = MUG(split='train', transforms=transforms,
                tab_embedder=TextEmbedder(cellwise=False, model='clip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_tab(trainset, batch_size=32)
    np.save('clip_tab_train.npy', embs)

    trainset = MUG(split='val', transforms=transforms,
                tab_embedder=TextEmbedder(cellwise=False, model='clip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_tab(trainset, batch_size=32)
    np.save('clip_tab_val.npy', embs)

    trainset = MUG(split='test', transforms=transforms, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_tab(trainset, batch_size=32)
    np.save('clip_tab_test.npy', embs) 
    
if __name__ == "__main__":
    text_cell()
    text_tab_clip()
    # text_gpt()
    text_cell_context()
    