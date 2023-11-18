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
from tabular_utils import OneHotEmbedder, TextEmbedder, Scarf, RandomMask, DefaultEmbedder
from dataset import DVM


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


def roberta_get_tab(dataset, batch_size):
    model = RobertaModel.from_pretrained("roberta-base")
    model = model.cuda()
    lines = []
    loader = DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size)
    for item in tqdm(loader):
        tab_line = item['tab_line']
        with torch.no_grad():
            model.eval()
            tab_line['input_ids'] = tab_line['input_ids'].squeeze(1).cuda()
            tab_line['attention_mask'] = tab_line['attention_mask'].squeeze(1).cuda()
            line = model(tab_line['input_ids'], tab_line['attention_mask'], return_dict=True)['pooler_output']
            lines.append(line.cpu().numpy())
    lines = np.concatenate(lines, axis=0)
    return lines    

def roberta_get_line(dataset, batch_size):
    model = RobertaModel.from_pretrained("roberta-base")
    model = model.cuda()
    lines = []
    loader = DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size)
    for item in tqdm(loader):
        tab_line = item['tab_line']
        b = tab_line['input_ids'].size(0)
        col_num = tab_line['input_ids'].size(1)
        with torch.no_grad():
            model.eval()
            tab_line['input_ids'] = tab_line['input_ids'].cuda()
            tab_line['input_ids'] = tab_line['input_ids'].flatten(0, 1)
            tab_line['attention_mask'] = tab_line['attention_mask'].cuda()
            tab_line['attention_mask'] = tab_line['attention_mask'].flatten(0, 1)
            line = model(tab_line['input_ids'], tab_line['attention_mask'], return_dict=True)['pooler_output']
            line = line.view((b, col_num, -1))
            lines.append(line.cpu().numpy())
    lines = np.concatenate(lines, axis=0)
    return lines    

def onehot_noaug():
    # onehot, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVM(split='train', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_noaug_train.npy', embs)

    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVM(split='val', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_noaug_val.npy', embs)

    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVM(split='test', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_noaug_test.npy', embs)

def onehot_scarf():
    # onehot, scarf augmentation
    tab_transform = Scarf(corrupt_rate=0.3)
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVM(split='train', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_scarf_train.npy', embs)


    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVM(split='val', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_scarf_val.npy', embs)

    trainset = DVM(split='test', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_scarf_test.npy', embs)

def text_cell():
    # text, cellwise, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = DVM(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_cell_train.npy', embs)

    trainset = DVM(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_cell_val.npy', embs)

    trainset = DVM(split='test', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_cell_test.npy', embs)

def text_cell_context():
    # text, cellwise, with context
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = DVM(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', context='car'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_context_train.npy', embs)

    trainset = DVM(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', context='car'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_context_val.npy', embs)

    trainset = DVM(split='test', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', context='car'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_context_test.npy', embs)

def text_gpt():
    # text, gpt template, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = DVM(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip', chatgpt_tmpl='/home/lmx/Image-Tabular-Fusion/data/dvm_car/chatgpt_tmpl.txt'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_gpt_train.npy', embs)

    trainset = DVM(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', chatgpt_tmpl='/home/lmx/Image-Tabular-Fusion/data/dvm_car/chatgpt_tmpl.txt'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_gpt_val.npy', embs)

    trainset = DVM(split='test', transforms=transforms, numerical=False, 
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
    trainset = DVM(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_tab(trainset, batch_size=32)
    np.save('clip_tab_train.npy', embs)

    trainset = DVM(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_tab(trainset, batch_size=32)
    np.save('clip_tab_val.npy', embs)

    trainset = DVM(split='test', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = clip_get_tab(trainset, batch_size=32)
    np.save('clip_tab_test.npy', embs)
    
    
def text_tab_roberta():
    # text, tabwise, roberta, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVM(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='roberta'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = roberta_get_tab(trainset, batch_size=32)
    np.save('roberta_tab_train.npy', embs)

    trainset = DVM(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='roberta'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = roberta_get_tab(trainset, batch_size=32)
    np.save('roberta_tab_val.npy', embs)

    trainset = DVM(split='test', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='roberta'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = roberta_get_tab(trainset, batch_size=32)
    np.save('roberta_tab_test.npy', embs)    

# def text_cell_roberta():
#     # text, tabwise, roberta, no augmentation
#     tab_transform = lambda x, y: y
#     transforms = {
#         'tab_tf': tab_transform
#     }
#     trainset = DVM(split='train', transforms=transforms, numerical=False, 
#                 tab_embedder=TextEmbedder(cellwise=True, model='roberta'), preload_images=False, 
#                 root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
#     embs = roberta_get_line(trainset, batch_size=32)
#     np.save('roberta_cell_train.npy', embs)

#     trainset = DVM(split='val', transforms=transforms, numerical=False, 
#                 tab_embedder=TextEmbedder(cellwise=True, model='roberta'), preload_images=False, 
#                 root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
#     embs = roberta_get_line(trainset, batch_size=32)
#     np.save('roberta_cell_val.npy', embs)

#     trainset = DVM(split='test', transforms=transforms, numerical=False, 
#                 tab_embedder=TextEmbedder(cellwise=True, model='roberta'), preload_images=False, 
#                 root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
#     embs = roberta_get_line(trainset, batch_size=32)
#     np.save('roberta_cell_test.npy', embs)    


def text_gpt_roberta():
    # text, tabwise, roberta, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVM(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='roberta', chatgpt_tmpl='/home/lmx/Image-Tabular-Fusion/data/dvm_car/chatgpt_tmpl.txt'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = roberta_get_line(trainset, batch_size=32)
    np.save('roberta_gpt_train.npy', embs)

    trainset = DVM(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='roberta', chatgpt_tmpl='/home/lmx/Image-Tabular-Fusion/data/dvm_car/chatgpt_tmpl.txt'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = roberta_get_line(trainset, batch_size=32)
    np.save('roberta_gpt_val.npy', embs)

    trainset = DVM(split='test', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='roberta', chatgpt_tmpl='/home/lmx/Image-Tabular-Fusion/data/dvm_car/chatgpt_tmpl.txt'), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = roberta_get_line(trainset, batch_size=32)
    np.save('roberta_gpt_test.npy', embs)   
    
def text_logit():
    # text, tabwise, roberta, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = DVM(split='train', transforms=transforms, numerical=True, 
                tab_embedder=DefaultEmbedder(), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('logit_train.npy', embs)

    trainset = DVM(split='val', transforms=transforms, numerical=True, 
                tab_embedder=DefaultEmbedder(), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('logit_val.npy', embs)
    
    trainset = DVM(split='test', transforms=transforms, numerical=True, 
                tab_embedder=DefaultEmbedder(), preload_images=False, 
                root_dir='/home/lmx/Image-Tabular-Fusion/data/dvm_car', modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('logit_test.npy', embs)


if __name__ == "__main__":
    # onehot_noaug()
    # onehot_scarf()
    # text_cell()
    # text_tab_clip()
    # text_tab_roberta()
    # text_gpt()
    # text_cell_context()
    # text_cell_roberta()
    # text_gpt_roberta()
    text_logit()