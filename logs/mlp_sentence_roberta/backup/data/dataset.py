from typing import Any
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from PIL import Image
import os.path as pth
import cv2
import torch

class MyData(Dataset):
    def __init__(self, split, transforms, root_dir, *args, **kwargs) -> None:
        super().__init__()
        self.split = split
        self.transforms = transforms
        self.root_dir = root_dir
    
    def __len__():
        pass
    
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    def get_labels(self):
        pass

class TabularData(MyData):
    def __init__(self, split, transforms, tab_embedder: None, root_dir:str='data/dvm_car', preload_images=False, *args, **kwargs) -> None:
        '''
        Args:
            split (str): data split, can be 'train', 'val', or 'test'
            transforms (dict): transforms on tabular data and image data, stored in a dictionary {'tab_tf': ..., 'img_tf': ...}
            tab_embedder (any): a generator encoding tabular values into embeddings 
        '''
        super().__init__(split, transforms, root_dir, *args, **kwargs)
        self.tab_embedder = tab_embedder
        self.preload_images = preload_images
    
    def _cache_images(self, paths):
        images = []
        for img_path in paths:
            _img = cv2.imread(img_path)
            if len(_img.shape) == 3:
                # if rgb
                _img = _img[...,[2,1,0]]
            elif len(_img.shape) == 2:
                # gray
                _img = np.expand_dims(_img, axis=-1)
            else:
                raise Exception(f"Incorrect image dimension {_img.shape}.")
            
            images.append(_img)
        images = np.concatenate(images, axis=-1)
        self.images = images
        
class DVM(TabularData):
    def __init__(self, split: str, transforms: dict, numerical: bool=False, 
                 tab_embedder=None, 
                 root_dir:str='data/dvm_car', preload_images=False, modal = ['img', 'tab'], *args, **kwargs) -> None:
        '''
            numerical (bool): whether use normalized and categorized numerical values for each cell
        '''
        super().__init__(split, transforms, tab_embedder, root_dir, preload_images, *args, **kwargs)
        self.split = split
        # tabular data
        self.df = pd.read_csv(pth.join(root_dir, f'{split}_df_full.csv'))
        self.labels = np.load(pth.join(root_dir, f'{split}_labels.npy'))
        
        assert len(self.df) == len(self.labels),'Label is inconsistent with the tabular data'
        
        # load tabular data meta infomation, including column field length and type
        with open(pth.join(root_dir, 'meta.json'), 'r') as f:
            self.meta_info = json.load(f)
        
        self.transforms = transforms
        
        if numerical:
            for c in list(self.meta_info.keys()):
                self.df[c] = self.df[c+'_num'].values
        
        self.tab_embedder = tab_embedder
        
        self.modal = modal
        if 'img' in modal:
            image_paths = np.load(pth.join(root_dir, f'{split}_paths.npy'))
            if preload_images:
                self.images = self._cache_images(image_paths)
            else:
                self.images = image_paths
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int) -> dict:
        
        data = {}
        
        if 'tab' in self.modal:
            # load tabular data
            tab_tf = self.transforms['tab_tf']
            df = tab_tf(index, self.df)
            line = self.tab_embedder.get_line(df, self.meta_info, index)
            tab_line = line['line_embd']
            data['tab_line'] = tab_line
            data['tab_sentence'] = line['line_sentence']
            
        if 'img' in self.modal:
            # load image data
            img_tf = self.transforms['img_tf']
            if self.preload_images:
                image = Image.fromarray(self.images[index])
            else:
                image = Image.open(self.images[index])
            image = img_tf(image).float()
            data['image'] = image
        
        label = self.labels[index]
        
        data['label'] = label
        
        return data
        
    def get_labels(self):
        # for resampling
        return self.labels
    
    def __repr__(self) -> str:
        return super().__repr__()


class DVMPre(Dataset):
    def __init__(self, split: str, transforms: dict, kwd:str='',
                 root_dir:str='data/dvm_car', modal=['img', 'tab'], preload_images=False, *args, **kwargs) -> None:
        super().__init__()
        self.split = split
        # tabular data
        self.df = pd.read_csv(pth.join(root_dir, f'{split}_df_full.csv'))
        self.labels = np.load(pth.join(root_dir, f'{split}_labels.npy'))
        self.tab_data = np.load(pth.join(root_dir, f"{kwd}_{split}.npy"))
        
        assert len(self.df) == len(self.labels),'Label is inconsistent with the tabular data'
        
        # load tabular data meta infomation, including column field length and type
        with open(pth.join(root_dir, 'meta.json'), 'r') as f:
            self.meta_info = json.load(f)
        
        self.transforms = transforms
        
        self.preload_images = preload_images
        self.modal = modal
        if 'img' in modal:
            image_paths = np.load(pth.join(root_dir, f'{split}_paths.npy'))
            if preload_images:
                self.images = self._cache_images(image_paths)
            else:
                self.images = image_paths    
                
                
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int) -> dict:
        
        data = {}
        
        if 'tab' in self.modal:
            # load tabular data
            data['tab_line'] = torch.tensor(self.tab_data[index, ...]).float()
            
        if 'img' in self.modal:
            # load image data
            img_tf = self.transforms['img_tf']
            if self.preload_images:
                image = Image.fromarray(self.images[index])
            else:
                image = Image.open(self.images[index])
            image = img_tf(image).float()
            data['image'] = image
        
        label = self.labels[index]
        
        data['label'] = label
        
        return data
        
    def get_labels(self):
        # for resampling
        return self.labels
    
    def __repr__(self) -> str:
        return super().__repr__()
    
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as T
    from .tabular_utils import OneHotEmbedder, DefaultEmbedder, Scarf, RandomMask
    
    tab_transform = Scarf(corrupt_rate=0.7)
    # tab_transform = RandomMask(corrupt_rate=0.7)
    
    transforms = {
        'tab_tf': tab_transform, 
        'img_tf': T.Compose(
            [
                T.ToTensor()
            ]
        )
    }
    
    # trainset = DVM(split='train', transforms=transforms, numerical=False, tab_embedder=DefaultEmbedder())
    # trainset = DVM(split='train', transforms=transforms, numerical=False, tab_embedder=TextEmbedder())
    trainset = DVM(split='train', transforms=transforms, numerical=True, tab_embedder=OneHotEmbedder())
    data = trainset[0]
    image = data['image']
    tab_line = data['tab_line']
    label = data['label']
    print(tab_line, tab_line.shape, label)
    plt.figure()
    plt.imshow(image.permute(1,2,0).numpy())
    plt.show()
    