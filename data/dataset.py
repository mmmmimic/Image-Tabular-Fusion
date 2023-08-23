from typing import Any
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json

class DVM(Dataset):
    def __init__(self, split: str, transforms: dict, numerical: bool=False) -> None:
        '''
        Args:
            split (str): data split, can be 'train', 'val', or 'test'
            transforms (dict): transforms on tabular data and image data, stored in a dictionary {'tab_tf': ..., 'img_tf': ...}
            numerical (bool): whether use normalized and categorized numerical values for each cell
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
        
        self.df = self.df[list(self.meta_info.keys())]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int) -> dict:
        data = {} # wrap up everything in a dictionary
        # load tabular data
        
        return {}
    
    def __repr__(self) -> str:
        return super().__repr__()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as T
    
    def table_transform(df):
        # onehot embedding
        pass
    transforms = {
        'tab_tf': None, 
        'img_tf': T.Compose(
            [
                T.ToTensor(),
            ]
        )
    }
    trainset = DVM(split='train', transforms=transforms, numerical=True)