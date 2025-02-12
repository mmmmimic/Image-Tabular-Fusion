'''
Preprocess of the skin dataset
'''
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
ROOT_PATH = 'data/skin'
# META_PATH = join(ROOT_PATH, 'metadata.csv')
TABLE_PATH = join(ROOT_PATH, 'release_midas.xlsx')
CATEGORICAL_IDS = ['midas_distance', 'midas_location', 'midas_gender', 
                    'midas_fitzpatrick', 'midas_ethnicity', 
                    'midas_race']
CONTINOUS_IDS = ['midas_age', 'length_(mm)', 'width_(mm)']
LABEL = 'midas_melanoma'
######################

# read tables and convert excel to csv
def read_csv(TABLE_PATH):
    df = pd.read_excel(TABLE_PATH)
    df.dropna(subset=CONTINOUS_IDS+[LABEL], inplace=True)
    df.fillna(dict(zip(CATEGORICAL_IDS, ['Unknown']*len(CATEGORICAL_IDS))), inplace=True)
    print(len(df))
    print(f'There are {len(np.unique(df.iloc[:,1]))} subjects.')
    # TABLE_PATH = TABLE_PATH.replace('.xlsx', '.csv')
    # table_df.to_csv (TABLE_PATH,  
    #                 index = None, 
    #                 header=True) 
    # table_df = pd.read_csv(TABLE_PATH)
    return df

# collect images
def collect_images(suffix='jpg'):
    imgs_dirs = glob(join(ROOT_PATH, f'*.{suffix}'))
    print(f'There are {len(imgs_dirs)} {suffix} images in the folder.')
    return imgs_dirs

def read_jpg(image_dir):
    img = cv2.imread(image_dir)[...,[2,1,0]]
    return img

def analyse_columns(df):
    columns = list(df.columns)
    for c in columns:
        print(c)
        print(Counter(df[c].values))

def process_columns(df):
    # categorical columns
    for c in CATEGORICAL_IDS:
        df[[c]] = df[[c]].astype('str')
        df[c] = df[c].map(lambda x: x.lower())
        
    # continous columns
    for c in CONTINOUS_IDS:
        df[c] = df[c].map(lambda x: np.float16(x))
    return df


def add_jitter(x, jitter=50):
    return x + random.randint(-jitter, jitter)

def add_jitter_float(x, jitter=10):
    return x + (round(np.random.rand(), 2)-0.5)*2*jitter
    
# wrapup all information in a table
def wrapup(table_df):
    subject_id = table_df['midas_record_id'].values 

    png_dirs = table_df['midas_file_name'].values
    png_dirs = list(map(lambda x: os.path.join(ROOT_PATH, x), png_dirs))
    labels = []
    image_ids = []
    subject_ids = []
    value_dict = {}
    
    for c in CATEGORICAL_IDS + CONTINOUS_IDS:
        value_dict[c] = []
    
    for i in tqdm(range(len(png_dirs))):
        image_id = png_dirs[i]
        image_ids.append(image_id)
        
        subject_id_ = subject_id[i]
        subject_ids.append(subject_id_)
        
        label = table_df[LABEL].values[i]
        label = 1 if label == 'yes' else 0
        labels.append(label)
        
        for c in CATEGORICAL_IDS + CONTINOUS_IDS:
            value_dict[c].append(table_df[c].values[i])
        
    value_dict['Subject_ID'] = subject_ids
    value_dict['label'] = labels
    value_dict['Image_ID'] = image_ids
    value_dict['Image_DIR'] = png_dirs
    
    value_dict = pd.DataFrame(value_dict)
    value_dict.to_csv(join(ROOT_PATH, 'data.csv'), index=False)

def postprocess():
    df = pd.read_csv(join(ROOT_PATH, 'data.csv'))
    # fix random seed
    seed = 2023
    np.random.seed(seed)
    random.seed(seed)
    # get meta info
    meta = {}
    for c in CATEGORICAL_IDS:
        df[c] = df[c].astype('category')
        field_length = len(list(set(df[c].values)))
        meta[c] = {
            'field_length': field_length,
            'type': 'categorical',
            'full_name': ''
        }
        values = list(set(df[c].values))
        df[c+'_num'] = df[c].values
        df[c+'_num'] = df[c+'_num'].replace(values, list(range(len(values))))
    
    # add noise
    df['midas_age'] = df['midas_age'].apply(add_jitter, args=(3,))
    df['midas_age'] = df['midas_age'].astype('int')
    df['length_(mm)'] = df['length_(mm)'].apply(add_jitter_float, args=(0.8,))
    df['width_(mm)'] = df['width_(mm)'].apply(add_jitter_float, args=(0.8,))
        
    # normalize
    for c in CONTINOUS_IDS:
        meta[c] = {
            'field_length': 1,
            'type': 'continuous',
            'full_name': ''
        }
        v = df[c].values
        df[c+'_num'] = (v - v.mean()) / (v.std())
    
    
    # train_val_test split
    train_ratio = 0.6
    test_ratio = 0.2 # val ratio = 0.1
    
    # for label == 0
    subject_id = np.array(list(set(df[df['label']==0]['Subject_ID'].values))) 
    indices = list(range(len(subject_id)))
    np.random.shuffle(indices)
    train_inds_ = indices[:int(train_ratio*len(subject_id))]
    test_inds_ = indices[int(train_ratio*len(subject_id)):int((train_ratio + test_ratio)*len(subject_id))]
    val_inds_ = indices[int((train_ratio + test_ratio)*len(subject_id)):]
    train_subject = list(subject_id[train_inds_])
    val_subject = list(subject_id[val_inds_])
    test_subject = list(subject_id[test_inds_])
    
    
    # for label == 1
    subject_id = np.array(list(set(df[df['label']==1]['Subject_ID'].values))) 
    indices = list(range(len(subject_id)))
    np.random.shuffle(indices)
    train_inds_ = indices[:int(train_ratio*len(subject_id))]
    test_inds_ = indices[int(train_ratio*len(subject_id)):int((train_ratio + test_ratio)*len(subject_id))]
    val_inds_ = indices[int((train_ratio + test_ratio)*len(subject_id)):]
    train_subject = list(subject_id[train_inds_]) + train_subject
    val_subject = list(subject_id[val_inds_]) + val_subject
    test_subject = list(subject_id[test_inds_]) + test_subject

    subject_id = np.array(subject_id)
    
    def split(x):
        if x in train_subject:
            return 'train'
        elif x in val_subject:
            return 'val'
        elif x in test_subject:
            return 'test'
        
    splits = np.array(list(map(split, df['Subject_ID'])))
    df['split'] = splits
    
    print(Counter(df[df['split']=='train']['label'].values))
    print(Counter(df[df['split']=='val']['label'].values))
    print(Counter(df[df['split']=='test']['label'].values))
    
    
    df.to_csv(join(ROOT_PATH, 'data_full.csv'), index=False)
    with open(join(ROOT_PATH, 'meta.json'), 'w') as f:
        f.write(json.dumps(meta, indent=4))


def check_image():
    ## remove images that do not exist
    df = pd.read_csv(join(ROOT_PATH, 'data_full.csv'))
    exist = []
    for i in range(len(df)):
        exist.append(os.path.exists(df['Image_DIR'].values[i]))
    exist = np.array(exist)
    df = df.iloc[exist==True, :]
    df.to_csv(join(ROOT_PATH, 'data_full.csv'), index=False)
    df = pd.read_csv(join(ROOT_PATH, 'data.csv'))
    df = df.iloc[exist==True, :]
    df.to_csv(join(ROOT_PATH, 'data.csv'), index=False)    


# def sample():
#     df = pd.read_csv(join(ROOT_PATH, 'data_full.csv'))
#     seed = 2023
#     random.seed(seed)
#     np.random.seed(seed)
#     # sample 10% train data
#     train_df = df[df['SPLIT']=='train']
#     val_df = df[df['SPLIT']=='val']
#     test_df = df[df['SPLIT']=='test']
    
#     n_samples = len(train_df)
#     inds =list(range(n_samples))
#     np.random.shuffle(inds)
#     inds = inds[:int(0.1*n_samples)]
#     np.save(join(ROOT_PATH, 'sample_ind.npy'), np.array(inds))
#     # train_df = train_df.reindex()
#     # train_df = train_df.iloc[inds,:]
#     # print(train_df)
#     # print(Counter(train_df['LABEL'].values))
#     # df = pd.concat((train_df, val_df, test_df), axis=0)
#     # df = df.reindex()
#     # df.to_csv(join(ROOT_PATH, 'data_sampled.csv'), index=False)
    
#     # print(inds)

# # def balance_sample():
# #     df = pd.read_csv(join(ROOT_PATH, 'data_full.csv'))
# #     seed = 2023
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     # sample 10% train data

# #     train_df = df[df['SPLIT']=='train']
# #     val_df = df[df['SPLIT']=='val']
# #     test_df = df[df['SPLIT']=='test']
    
# #     # train
# #     inds = []
# #     num = 1
# #     for s in list(set(train_df['SUBJECT ID'].values)):
# #         inds.append(train_df[train_df['SUBJECT ID']==s].index.values[:num])
# #     # for s in list(set(val_df['SUBJECT ID'].values)):
# #     #     inds.append(val_df[val_df['SUBJECT ID']==s].index.values[:num])    
# #     # for s in list(set(test_df['SUBJECT ID'].values)):
# #     #     inds.append(test_df[test_df['SUBJECT ID']==s].index.values[:num])
# #     inds = np.concatenate(inds, axis=0)
# #     n_samples = len(inds)
# #     print(n_samples)
    
# #     print(inds)
    
# #     df = df.iloc[inds,:]
# #     print(df)
# #     print(Counter(df['SUBJECT ID'].values))
# #     print(Counter(df['SPLIT'].values))
# #     print(Counter(df['LABEL'].values))
    
# #     # np.save(join(ROOT_PATH, 'balance_ind.npy'), np.array(inds))


if __name__ == "__main__":
    table_df = read_csv(TABLE_PATH)
    jpg_dirs = collect_images(suffix='jpg') + collect_images(suffix='jpeg')
    print(len(jpg_dirs))
    # analyse_columns(table_df)
    processed_table_df = process_columns(table_df)
    print(processed_table_df)
    wrapup(processed_table_df)
    postprocess()
    # sample()
    check_image()
    