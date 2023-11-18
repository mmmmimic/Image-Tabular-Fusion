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
    label_key = 'type_1'
    
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
    def _get_labels(train_df, dev_df, test_df):
        df = pd.concat([train_df, dev_df, test_df], axis=0)
        labels = df[label_key].values
        unique_labels = list(set(labels))
        mapping_dict = dict(zip(unique_labels, list(range(len(unique_labels)))))
        return mapping_dict
    
    mapping_dict = _get_labels(train_df, dev_df, test_df)
    
    labels = train_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    train_df['label'] = labels
    
    labels = dev_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    dev_df['label'] = labels
    
    labels = test_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    test_df['label'] = labels
    
    # save df
    train_df.to_csv(join(ROOT_PATH, 'train_full.csv'), index=False)
    dev_df.to_csv(join(ROOT_PATH, 'dev_full.csv'), index=False)
    test_df.to_csv(join(ROOT_PATH, 'test_full.csv'), index=False)
    
    # create meta.json
    meta = {}
    for c in columns:
        if c not in ['Image Path', label_key]:
            meta[c] = {
                        'field_length': 1,
                        'type': 'continuous',
                        'full_name': ''
                    }
    # with open(join(ROOT_PATH, 'meta.json'), 'w') as f:
    #     f.write(json.dumps(meta, indent=4))            

def process_pokemon_secondarytype(ROOT_PATH):
    ROOT_PATH = join(ROOT_PATH, 'Pokemon-secondary_type')
    label_key = 'type_2'
    
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
    def _get_labels(train_df, dev_df, test_df):
        df = pd.concat([train_df, dev_df, test_df], axis=0)
        labels = df[label_key].values
        unique_labels = list(set(labels))
        mapping_dict = dict(zip(unique_labels, list(range(len(unique_labels)))))
        return mapping_dict
    
    mapping_dict = _get_labels(train_df, dev_df, test_df)
    
    labels = train_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    train_df['label'] = labels
    
    labels = dev_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    dev_df['label'] = labels
    
    labels = test_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    test_df['label'] = labels
    
    # save df
    train_df.to_csv(join(ROOT_PATH, 'train_full.csv'), index=False)
    dev_df.to_csv(join(ROOT_PATH, 'dev_full.csv'), index=False)
    test_df.to_csv(join(ROOT_PATH, 'test_full.csv'), index=False)
    
    # create meta.json
    meta = {}
    for c in columns:
        if c not in ['Image Path', label_key]:
            meta[c] = {
                        'field_length': 1,
                        'type': 'continuous',
                        'full_name': ''
                    }
    # with open(join(ROOT_PATH, 'meta.json'), 'w') as f:
    #     f.write(json.dumps(meta, indent=4))         

def process_lol_sc(ROOT_PATH):
    ROOT_PATH = join(ROOT_PATH, 'LeagueOfLegends-Skin-category')
    label_key = 'Category'
    
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
    def _get_labels(train_df, dev_df, test_df):
        df = pd.concat([train_df, dev_df, test_df], axis=0)
        labels = df[label_key].values
        unique_labels = list(set(labels))
        mapping_dict = dict(zip(unique_labels, list(range(len(unique_labels)))))
        return mapping_dict
    
    mapping_dict = _get_labels(train_df, dev_df, test_df)
    
    labels = train_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    train_df['label'] = labels
    
    labels = dev_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    dev_df['label'] = labels
    
    labels = test_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    test_df['label'] = labels
    
    # save df
    train_df.to_csv(join(ROOT_PATH, 'train_full.csv'), index=False)
    dev_df.to_csv(join(ROOT_PATH, 'dev_full.csv'), index=False)
    test_df.to_csv(join(ROOT_PATH, 'test_full.csv'), index=False)
    
    # create meta.json
    meta = {}
    for c in columns:
        if c not in ['Image Path', label_key]:
            meta[c] = {
                        'field_length': 1,
                        'type': 'continuous',
                        'full_name': ''
                    }
    # with open(join(ROOT_PATH, 'meta.json'), 'w') as f:
    #     f.write(json.dumps(meta, indent=4))        

def process_csg_sq(ROOT_PATH):
    ROOT_PATH = join(ROOT_PATH, 'CSGO-Skin-quality')
    label_key = 'Skin Quality'
    
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
    def _get_labels(train_df, dev_df, test_df):
        df = pd.concat([train_df, dev_df, test_df], axis=0)
        labels = df[label_key].values
        unique_labels = list(set(labels))
        mapping_dict = dict(zip(unique_labels, list(range(len(unique_labels)))))
        return mapping_dict
    
    mapping_dict = _get_labels(train_df, dev_df, test_df)
    
    labels = train_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    train_df['label'] = labels
    
    labels = dev_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    dev_df['label'] = labels
    
    labels = test_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    test_df['label'] = labels
    
    # save df
    train_df.to_csv(join(ROOT_PATH, 'train_full.csv'), index=False)
    dev_df.to_csv(join(ROOT_PATH, 'dev_full.csv'), index=False)
    test_df.to_csv(join(ROOT_PATH, 'test_full.csv'), index=False)
    
    # create meta.json
    meta = {}
    for c in columns:
        if c not in ['Image Path', label_key]:
            meta[c] = {
                        'field_length': 1,
                        'type': 'continuous',
                        'full_name': ''
                    }
    # with open(join(ROOT_PATH, 'meta.json'), 'w') as f:
    #     f.write(json.dumps(meta, indent=4))      

def process_hs_ac(ROOT_PATH):
    ROOT_PATH = join(ROOT_PATH, 'Hearthstone-All-cardClass')
    label_key = 'cardClass'
    
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
    def _get_labels(train_df, dev_df, test_df):
        df = pd.concat([train_df, dev_df, test_df], axis=0)
        labels = df[label_key].values
        unique_labels = list(set(labels))
        mapping_dict = dict(zip(unique_labels, list(range(len(unique_labels)))))
        return mapping_dict
    
    mapping_dict = _get_labels(train_df, dev_df, test_df)
    
    labels = train_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    train_df['label'] = labels
    
    labels = dev_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    dev_df['label'] = labels
    
    labels = test_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    test_df['label'] = labels
    
    # save df
    train_df.to_csv(join(ROOT_PATH, 'train_full.csv'), index=False)
    dev_df.to_csv(join(ROOT_PATH, 'dev_full.csv'), index=False)
    test_df.to_csv(join(ROOT_PATH, 'test_full.csv'), index=False)
    
    # create meta.json
    meta = {}
    for c in columns:
        if c not in ['Image Path', label_key]:
            meta[c] = {
                        'field_length': 1,
                        'type': 'continuous',
                        'full_name': ''
                    }
    # with open(join(ROOT_PATH, 'meta.json'), 'w') as f:
    #     f.write(json.dumps(meta, indent=4))    

def process_hs_as(ROOT_PATH):
    ROOT_PATH = join(ROOT_PATH, 'Hearthstone-All-set')
    label_key = 'set'
    
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
    def _get_labels(train_df, dev_df, test_df):
        df = pd.concat([train_df, dev_df, test_df], axis=0)
        labels = df[label_key].values
        unique_labels = list(set(labels))
        mapping_dict = dict(zip(unique_labels, list(range(len(unique_labels)))))
        return mapping_dict
    
    mapping_dict = _get_labels(train_df, dev_df, test_df)
    
    labels = train_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    train_df['label'] = labels
    
    labels = dev_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    dev_df['label'] = labels
    
    labels = test_df[label_key].values    
    labels = np.array(list(map(lambda x: mapping_dict[x], labels)), dtype=np.int32)
    test_df['label'] = labels
    
    # save df
    train_df.to_csv(join(ROOT_PATH, 'train_full.csv'), index=False)
    dev_df.to_csv(join(ROOT_PATH, 'dev_full.csv'), index=False)
    test_df.to_csv(join(ROOT_PATH, 'test_full.csv'), index=False)
    
    # create meta.json
    meta = {}
    for c in columns:
        if c not in ['Image Path', label_key]:
            meta[c] = {
                        'field_length': 1,
                        'type': 'continuous',
                        'full_name': ''
                    }
    # with open(join(ROOT_PATH, 'meta.json'), 'w') as f:
    #     f.write(json.dumps(meta, indent=4))   
        
        
if __name__ == "__main__":
    process_pokemon_primarytype(ROOT_PATH)
    process_pokemon_secondarytype(ROOT_PATH)
    process_lol_sc(ROOT_PATH)
    process_csg_sq(ROOT_PATH)
    process_hs_ac(ROOT_PATH)
    process_hs_as(ROOT_PATH)
    