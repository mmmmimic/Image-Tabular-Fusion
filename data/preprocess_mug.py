import pydicom
import numpy as np
import cv2
from PIL import Image
from pathlib import PurePath
import pandas as pd
from tqdm import tqdm
from glob import glob
from collections import Counter
import os
from os.path import join
import random
import json

##### CONSTANTS ######
ROOT_PATH = 'data/mug_dataset'
##############################

def process_pokemon_primarytype(ROOT_PATH):
    ROOT_PATH = join(ROOT_PATH, 'Pokemon-primary_type')
    train_df = pd.read_csv(join(ROOT_PATH, 'train.csv'))
    dev_df = pd.read_csv(join(ROOT_PATH, 'dev.csv'))
    test_df = pd.read_csv(join(ROOT_PATH, 'test.csv'))
    
    columns = list(train_df.columns)
    
    print(columns)
    
    # fill the table
    def _fill(df):
        df = df.astype('string')
        df = df.replace(np.nan, 'Unkonwn')
        return df

    train_df = _fill(train_df)
    dev_df = _fill(dev_df)
    test_df = _fill(test_df)
    
    # save image path and save label
    def _get_labels(df):
        labels = df['type_1'].values
        unique_labels = list(set(labels))
        mapping_dict = dict(zip(unique_labels, list(range(len(unique_labels)))))
        labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
        df['label'] = labels
        return df

    train_df = _get_labels(train_df)
    dev_df = _get_labels(dev_df)
    test_df = _get_labels(test_df)
    
    # save df
    train_df.to_csv(join(ROOT_PATH, 'train_full.csv'), index=False)
    dev_df.to_csv(join(ROOT_PATH, 'dev_full.csv'), index=False)
    test_df.to_csv(join(ROOT_PATH, 'test_full.csv'), index=False)
    
    # create meta.json
    meta = {}
    for c in columns:
        if c not in ['Image Path', 'type_1']:
            meta[c] = {
                        'field_length': 1,
                        'type': 'continuous',
                        'full_name': ''
                    }
    with open(join(ROOT_PATH, 'meta.json'), 'w') as f:
        f.write(json.dumps(meta, indent=4))            
    
if __name__ == "__main__":
    process_pokemon_primarytype(ROOT_PATH)
    