import numpy as np
import torch
import clip
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from medclip import MedCLIPModel, MedCLIPVisionModel


# add embedder into path
sys.path.append('/home/lmx/Image-Tabular-Fusion/data')
from tabular_utils import OneHotEmbedder, TextEmbedder
from dataset import COVID19AR
from os.path import join

ROOT = '/home/lmx/Image-Tabular-Fusion/data/covid19_ar'

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

def medclip_get_line(dataset, batch_size):
    lines = []
    loader = DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size)
    model = MedCLIPModel(vision_cls = MedCLIPVisionModel)
    model.from_pretrained()
    model = model.text_model.cuda()
    
    for item in tqdm(loader):
        tab_line = item['tab_line']
        b = tab_line['input_ids'].size(0)
        col_num = tab_line['input_ids'].size(1)
        tab_line['input_ids'] = tab_line['input_ids'].flatten(0, 1).cuda()
        tab_line['attention_mask'] = tab_line['attention_mask'].flatten(0, 1).cuda()
        with torch.no_grad():
            line = model(input_ids=tab_line['input_ids'], attention_mask=tab_line['attention_mask'])
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

def onehot_noaug():
    # onehot, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = COVID19AR(split='train', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_noaug_train.npy', embs)

    transforms = {
        'tab_tf': tab_transform
    }
    trainset = COVID19AR(split='val', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_noaug_val.npy', embs)

    transforms = {
        'tab_tf': tab_transform
    }
    trainset = COVID19AR(split='test', transforms=transforms, numerical=True, 
                tab_embedder=OneHotEmbedder(), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = get_all(trainset, batch_size=32)
    np.save('onehot_noaug_test.npy', embs)

def text_cell():
    # text, cellwise, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = COVID19AR(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_cell_train.npy', embs)

    trainset = COVID19AR(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_cell_val.npy', embs)

    trainset = COVID19AR(split='test', transforms=transforms, numerical=False, 
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
    trainset = COVID19AR(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', context='covid19 patient'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_context_train.npy', embs)

    trainset = COVID19AR(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', context='covid19 patient'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_context_val.npy', embs)

    trainset = COVID19AR(split='test', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', context='covid19 patient'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_context_test.npy', embs)

def text_gpt():
    # text, gpt template, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = COVID19AR(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip', chatgpt_tmpl=join(ROOT, 'chatgpt_tmpl.txt')), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_gpt_train.npy', embs)

    trainset = COVID19AR(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', chatgpt_tmpl=join(ROOT, 'chatgpt_tmpl.txt')), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_gpt_val.npy', embs)

    trainset = COVID19AR(split='test', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='clip', chatgpt_tmpl=join(ROOT, 'chatgpt_tmpl.txt')), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_line(trainset, batch_size=32)
    np.save('clip_gpt_test.npy', embs)

def text_tab_clip():
    # text, tabwise, clip, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    }
    trainset = COVID19AR(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_tab(trainset, batch_size=32)
    np.save('clip_tab_train.npy', embs)

    trainset = COVID19AR(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_tab(trainset, batch_size=32)
    np.save('clip_tab_val.npy', embs)

    trainset = COVID19AR(split='test', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=False, model='clip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = clip_get_tab(trainset, batch_size=32)
    np.save('clip_tab_test.npy', embs)


def text_medclip():
    # text, cellwise, no augmentation
    tab_transform = lambda x, y: y
    transforms = {
        'tab_tf': tab_transform
    } 
    trainset = COVID19AR(split='train', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='medclip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = medclip_get_line(trainset, batch_size=32)
    np.save('medclip_cell_train.npy', embs)

    trainset = COVID19AR(split='val', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='medclip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = medclip_get_line(trainset, batch_size=32)
    np.save('medclip_cell_val.npy', embs)

    trainset = COVID19AR(split='test', transforms=transforms, numerical=False, 
                tab_embedder=TextEmbedder(cellwise=True, model='medclip'), preload_images=False, 
                root_dir=ROOT, modal=['tab'])
    embs = medclip_get_line(trainset, batch_size=32)
    np.save('medclip_cell_test.npy', embs)

if __name__ == "__main__":
    # onehot_noaug()
    # text_cell()
    # text_tab_clip()
    # text_gpt()
    # text_cell_context()
    text_medclip()
    