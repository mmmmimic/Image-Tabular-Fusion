from typing import Any
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from PIL import Image
from tabular_tools import OneHotEmbedder, DefaultEmbedder, Scarf

class DVM(Dataset):
    def __init__(self, split: str, transforms: dict, numerical: bool=False, tab_embedder: Any=DefaultEmbedder) -> None:
        '''
        Args:
            split (str): data split, can be 'train', 'val', or 'test'
            transforms (dict): transforms on tabular data and image data, stored in a dictionary {'tab_tf': ..., 'img_tf': ...}
            numerical (bool): whether use normalized and categorized numerical values for each cell
            tab_embedder (any): a generator encoding tabular values into embeddings 
        '''
        super(DVM).__init__()
        self.split = split
        # tabular data
        self.df = pd.read_csv(f'data/dvm_car/{split}_df_full.csv')
        self.labels = np.load(f'data/dvm_car/{split}_labels.npy')
        assert len(self.df) == len(self.labels),'Label is inconsistent with the tabular data'
        # load tabular data meta infomation, including column field length and type
        with open('data/dvm_car/meta.json', 'r') as f:
            self.meta_info = json.load(f)
        
        self.transforms = transforms
        
        if numerical:
            for c in list(self.meta_info.keys()):
                self.df[c] = self.df[c+'_num'].values
        
        self.tab_embedder = tab_embedder
        
        self.image_paths = np.load(f'data/dvm_car/{split}_paths.npy')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int) -> dict:
        tab_tf, img_tf = self.transforms['tab_tf'], self.transforms['img_tf']
        # load tabular data
        df = tab_tf(index, self.df)
        tab_line = self.tab_embedder.get_line(df, self.meta_info, index)

        # load image data
        image = Image.open(self.image_paths[index])
        image = img_tf(image)
        
        return { # wrap up everything in a dictionary
                'image': image,
                'tab_line': tab_line
                }
    
    def __repr__(self) -> str:
        return super().__repr__()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as T
    
    tab_transform = Scarf(corrupt_rate=0.7)
    
    transforms = {
        'tab_tf': tab_transform, 
        'img_tf': T.Compose(
            [
                T.ToTensor()
            ]
        )
    }
    # trainset = DVM(split='train', transforms=transforms, numerical=False, tab_embedder=DefaultEmbedder())
    trainset = DVM(split='train', transforms=transforms, numerical=True, tab_embedder=OneHotEmbedder())
    data = trainset[100]
    image = data['image']
    tab_line = data['tab_line']
    print(tab_line, tab_line.shape)
    plt.figure()
    plt.imshow(image.permute(1,2,0).numpy())
    plt.show()