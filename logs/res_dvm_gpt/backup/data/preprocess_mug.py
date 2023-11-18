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

def process_hs_mr(ROOT_PATH):
    ROOT_PATH = join(ROOT_PATH, 'Hearthstone-Minion-race')
    label_key = 'race'
    
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
    with open(join(ROOT_PATH, 'meta.json'), 'w') as f:
        f.write(json.dumps(meta, indent=4))   
            
def process_hs_ss(ROOT_PATH):
    ROOT_PATH = join(ROOT_PATH, 'Hearthstone-Spell-spellSchool')
    label_key = 'spellSchool'
    
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
    with open(join(ROOT_PATH, 'meta.json'), 'w') as f:
        f.write(json.dumps(meta, indent=4))   

def inspect_column(column):
    if column.dtype == 'float64' or column.dtype == 'int64':
        return 'continuous'
    else:
        c = Counter(column)
        if len(c.keys()) > 100:
            return 'textual'
        else:
            return 'categorical'
    
def extract(ROOT_PATH, token='Pokemon-primary_type', label_key = 'type_1'):
    ROOT_PATH = join(ROOT_PATH, token)
    
    train_df = pd.read_csv(join(ROOT_PATH, 'train.csv'))
    dev_df = pd.read_csv(join(ROOT_PATH, 'dev.csv'))
    test_df = pd.read_csv(join(ROOT_PATH, 'test.csv'))
    
    # train_df.dropna(inplace=True, axis=0, how='any')
    # dev_df.dropna(inplace=True, axis=0, how='any')
    # test_df.dropna(inplace=True, axis=0, how='any')
    def _fill(df):
        columns = list(df.columns)        
        for c in columns:
            if c not in ['Image Path', label_key]:
                if df[c].dtype == 'object':
                    df[c].replace(np.nan, 'Unknown') # add an None Type, following the original dataset paper
                else:
                    df[c].replace(np.nan, 0)
        return df

    if token == 'CSGO-Skin-quality':
        train_df['Min Price'] = train_df['Min Price'].apply(lambda x: float(x[1:].replace(',','').replace('$', '').replace('/', '')))
        dev_df['Min Price'] = dev_df['Min Price'].apply(lambda x: float(x[1:].replace(',','').replace('$', '').replace('/', '')))
        test_df['Min Price'] = test_df['Min Price'].apply(lambda x: float(x[1:].replace(',','').replace('$', '').replace('/', '')))
        train_df['Max Price'] = train_df['Max Price'].apply(lambda x: float(x[1:].replace(',','').replace('$', '').replace('/', '')))
        dev_df['Max Price'] = dev_df['Max Price'].apply(lambda x: float(x[1:].replace(',','').replace('$', '').replace('/', '')))
        test_df['Max Price'] = test_df['Max Price'].apply(lambda x: float(x[1:].replace(',','').replace('$', '').replace('/', '')))
        train_df['Min Price'] = train_df['Min Price'].astype('float64')
        dev_df['Min Price'] = dev_df['Min Price'].astype('float64')
        test_df['Min Price'] = test_df['Min Price'].astype('float64')
        train_df['Max Price'] = train_df['Max Price'].astype('float64')
        dev_df['Max Price'] = dev_df['Max Price'].astype('float64')
        test_df['Max Price'] = test_df['Max Price'].astype('float64')
        
    train_df = _fill(train_df)
    dev_df = _fill(dev_df)
    test_df = _fill(test_df)
    
    
    df = pd.concat((train_df, dev_df, test_df), axis=0)
    
    columns = list(train_df.columns)
    
    cc = []
    meta = {}
    for c in columns:
        if c not in ['Image Path', label_key]:
            dtype = inspect_column(df[c])
            cc.append(dtype)
            if dtype == 'categorical':
                cnter = Counter(df[c].values)
                fl = len(cnter.keys())
                m_d = dict(zip(list(cnter.keys()), list(range(fl))))
                train_df[c+'_num'] = train_df[c].apply(lambda x: m_d[x])
                dev_df[c+'_num'] = dev_df[c].apply(lambda x: m_d[x])
                test_df[c+'_num'] = test_df[c].apply(lambda x: m_d[x])
                
            elif dtype == 'continuous':
                # did not normalize here, since normalization cause some NaN values
                fl = 1
                v = train_df[c].values
                train_df[c+'_num'] = v
                v = dev_df[c].values
                dev_df[c+'_num'] = v
                v = test_df[c].values
                test_df[c+'_num'] = v
                
            else:
                fl = 1
                
            meta[c] = {
                'field_length': fl,
                'type': dtype, 
                'full_name': c
            }
    cnter = Counter(cc)
    print(cnter['continuous'] + cnter['categorical'], cnter['textual'])
    # create meta.json
    with open(join(ROOT_PATH, 'meta_tri.json'), 'w') as f:
        f.write(json.dumps(meta, indent=4)) 
        
        
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
    
    train_df.to_csv(join(ROOT_PATH, 'train_tri.csv'), index=False)
    dev_df.to_csv(join(ROOT_PATH, 'dev_tri.csv'), index=False)
    test_df.to_csv(join(ROOT_PATH, 'test_tri.csv'), index=False)
    
       
                  
if __name__ == "__main__":
    # process_pokemon_primarytype(ROOT_PATH)
    # process_pokemon_secondarytype(ROOT_PATH)
    # process_lol_sc(ROOT_PATH)
    # process_csg_sq(ROOT_PATH)
    # process_hs_ac(ROOT_PATH)
    # process_hs_as(ROOT_PATH)
    # process_hs_mr(ROOT_PATH)
    # process_hs_ss(ROOT_PATH)
    extract(ROOT_PATH, 'Pokemon-primary_type', 'type_1')
    extract(ROOT_PATH, 'Pokemon-secondary_type', 'type_2')
    extract(ROOT_PATH, 'Hearthstone-All-cardClass', 'cardClass')
    extract(ROOT_PATH, 'Hearthstone-All-set', 'set')
    extract(ROOT_PATH, 'Hearthstone-Minion-race', 'race')
    extract(ROOT_PATH, 'Hearthstone-Spell-spellSchool', 'spellSchool')
    extract(ROOT_PATH, 'LeagueOfLegends-Skin-category', 'Category')
    extract(ROOT_PATH, 'CSGO-Skin-quality', 'Skin Quality')
    
    