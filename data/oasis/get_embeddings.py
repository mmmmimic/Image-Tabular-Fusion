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
import struct

# add embedder into path
sys.path.append('..')
from tabular_utils import OneHotEmbedder, TextEmbedder, Scarf, RandomMask, DefaultEmbedder
from dataset import Oasis

root_dir = '../oasis'

def get_all(dataset, batch_size):
    lines = []
    loader = DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size)
    for item in tqdm(loader):
        tab_line = item['tab_line']
        lines.append(tab_line.numpy())
    lines = np.concatenate(lines, axis=0)
    return lines
    
def onehot_noaug():
    # onehot, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = Oasis(split='train', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir=root_dir, modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_noaug_train.npy', embs)

    transforms = {
        'tab_tf': tab_transform
    }
    trainset = Oasis(split='val', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir=root_dir, modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_noaug_val.npy', embs)

    transforms = {
        'tab_tf': tab_transform
    }
    trainset = Oasis(split='test', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir=root_dir, modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_noaug_test.npy', embs)

def float_to_fixed_bin_array(num, bit_length=32):
    # 将浮动点数转换为二进制表示
    packed = struct.pack('!f', num)  # '!f'表示将浮动点数按大端格式打包为4字节
    # 转换为二进制字符串并去掉 '0b' 前缀，确保固定长度
    binary_str = ''.join(f'{byte:08b}' for byte in packed)
    
    # 截断或填充至目标长度
    binary_str = binary_str[:bit_length].ljust(bit_length, '0')
    
    # 将二进制字符串转为数组形式
    arr = np.array([int(bit) for bit in binary_str], dtype=np.float32)
    return torch.Tensor(arr).unsqueeze(1)

def text_logit():
    # text, tabwise, roberta, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = Oasis(split='train', transforms=transforms, numerical=True, 
                tab_embedder=DefaultEmbedder(), preload_images=False, 
                root_dir=root_dir, modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    # new_emb = []
    # for i in range(embs.shape[0]):
    #     tmp = []
    #     for j in range(embs.shape[1]):
    #         tmp.append(float_to_fixed_bin_array(embs[i,j]))
    #     tmp = torch.cat(tmp, dim=-1).unsqueeze(0)
    #     new_emb.append(tmp)
    # embs = torch.cat(new_emb, dim=0)
    np.save('logit_train.npy', embs)

    trainset = Oasis(split='val', transforms=transforms, numerical=True, 
                tab_embedder=DefaultEmbedder(), preload_images=False, 
                root_dir=root_dir, modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    # new_emb = []
    # for i in range(embs.shape[0]):
    #     tmp = []
    #     for j in range(embs.shape[1]):
    #         tmp.append(float_to_fixed_bin_array(embs[i,j]))
    #     tmp = torch.cat(tmp, dim=-1).unsqueeze(0)
    #     new_emb.append(tmp)
    # embs = torch.cat(new_emb, dim=0)
    np.save('logit_val.npy', embs)
    
    trainset = Oasis(split='test', transforms=transforms, numerical=True, 
                tab_embedder=DefaultEmbedder(), preload_images=False, 
                root_dir=root_dir, modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    # new_emb = []
    # for i in range(embs.shape[0]):
    #     tmp = []
    #     for j in range(embs.shape[1]):
    #         tmp.append(float_to_fixed_bin_array(embs[i,j]))
    #     tmp = torch.cat(tmp, dim=-1).unsqueeze(0)
    #     new_emb.append(tmp)
    # embs = torch.cat(new_emb, dim=0)
    np.save('logit_test.npy', embs)

if __name__ == "__main__":
    # onehot_noaug()
    text_logit()