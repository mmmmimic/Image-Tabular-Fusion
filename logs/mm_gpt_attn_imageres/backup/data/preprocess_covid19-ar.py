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
ROOT_PATH = 'data/covid19_ar'
META_PATH = join(ROOT_PATH, 'manifest-1594658036421', 'metadata.csv')
TABLE_PATH = join(ROOT_PATH, 'COVID-19 AR Clinical Correlates July202020.xlsx')
IMAGE_PATH = join(ROOT_PATH, 'manifest-1594658036421', 'COVID-19-AR')
CATEGORICAL_IDS = ['SEX', 'RACE', 'ZIP', 'EXTENSIVE BURNS', 'MALNUTRITION', 'CURRENT PREGNANT', 
                   'CHRONIC KIDNEY DISEASE', 'DIABETES TYPE I', 'DIABETES TYPE II', 'TRANSPLANT', 
                   'HEMODIALYSIS Pre Diagnosis', 'CANCER']
CONTINOUS_IDS = ['AGE', 'LATEST_BMI', 'LATEST WEIGHT', 'LATEST HEIGHT']
LABEL = 'ICU Admit'
######################

# read tables and convert excel to csv
def read_csv(META_PATH, TABLE_PATH):
    meta_df = pd.read_csv(META_PATH)
    table_df = pd.read_excel(TABLE_PATH, skiprows=1)
    TABLE_PATH = TABLE_PATH.replace('.xlsx', '.csv')
    table_df.to_csv (TABLE_PATH,  
                    index = None, 
                    header=True) 
    table_df = pd.read_csv(TABLE_PATH)
    print(f'There are {len(table_df)} subjects.')
    return meta_df, table_df

# collect images
def collect_images(IMAGE_PATH, suffix='dcm'):
    imgs_dirs = glob(join(IMAGE_PATH, '*', '*', '*', f'*.{suffix}'))
    print(f'There are {len(imgs_dirs)} {suffix} images in the folder.')
    return imgs_dirs

def read_dicom(dcm_dir):
    # read dicom images and convert them to *.png
    dcm_img = pydicom.dcmread(dcm_dir)
    img = dcm_img.pixel_array
    # uint16 -> uint8
    img = np.asarray((img / img.max())*255., dtype=np.uint8)
    return img
    
def save_img(img, img_dir):
    # convert gray image to 'rgb' image and save it
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
        img = np.concatenate((img, img, img), axis=-1)
    elif (len(img.shape) == 3):
        if img.shape[-1] == 1:
            img = np.concatenate((img, img, img), axis=-1)
        elif img.shape[-1] == 3:
            img = img[...,[2,1,0]]
            print(img_dir)
        else:
            raise ValueError()
    else:
        raise ValueError()
    cv2.imwrite(img_dir, img)

def iterate_on_images(dcm_dirs):
    for dcm_dir in tqdm(dcm_dirs):
        img_dir = dcm_dir.replace('.dcm', '.png')
        img = read_dicom(dcm_dir)
        save_img(img, img_dir)



# "Engine_size": {
#     "field_length": 1,
#     "type": "continuous",
#     "full_name": "engine size"
# },
# "Color": {
#     "field_length": 22,
#     "type": "categorical",
#     "full_name": "color"
# },

def analyse_columns(df):
    columns = list(df.columns)
    for c in columns:
        print(c)
        print(Counter(df[c].values))
        # print(len(df[df[c]=='Y'][df['ICU Admit']=='Y']), len(df[df[c]=='Y']), len(df[df['ICU Admit']=='Y']))
        # print(len(df[df[c]=='Y'][df['# ICU admits']==2]), len(df[df[c]=='Y']), len(df[df['# ICU admits']==2]))
        # print(len(df[df[c]=='Y'][df['# ICU admits']==3]), len(df[df[c]=='Y']), len(df[df['# ICU admits']==3]))
        # print(len(df[df[c]=='Y'][df['# ICU admits']==4]), len(df[df[c]=='Y']), len(df[df['# ICU admits']==4]))
        # print(len(df[df[c]=='Y'][df['MORTALITY']=='Y']), len(df[df[c]=='Y']), len(df[df['MORTALITY']=='Y']))        
        # print('--------------------')

def process_columns(df):
    # categorical columns
    df = df.dropna(subset='LATEST HEIGHT')
    for c in CATEGORICAL_IDS:
        df[[c]] = df[[c]].astype('str')
        df = df.replace('M', 'male')
        df = df.replace('F', 'female')
        df = df.replace('Y', 'yes')
        df = df.replace('N', 'no')
        
        cnter = Counter(df[c].values)
        for k, v in zip(cnter.keys(), cnter.values()):
            if v == 1:
                df = df.replace(k, 'other')
        df[c] = df[c].map(lambda x: x.lower())
        
    # continous columns
    for c in CONTINOUS_IDS:
        if c == 'LATEST HEIGHT':
            df[c] = df[c].map(lambda x: int(x.split('\'')[0])*30.48 + int(x[-2])*2.54)
        df[c] = df[c].map(lambda x: np.float16(x))
    return df


def add_jitter(x, jitter=50):
    return x + random.randint(-jitter, jitter)

# wrapup all information in a table
def wrapup(IMAGE_PATH, table_df, png_dirs):
    # fix random seed
    seed = 2023
    np.random.seed(seed)
    random.seed(seed)

    subject_id = table_df['PATIENT_ID'].values
    # train_val_test split
    train_ratio = 0.4
    val_ratio = 0.1 # test ratio = 0.5
    indices = list(range(len(subject_id)))
    np.random.shuffle(indices)
    train_inds = indices[:int(train_ratio*len(subject_id))]
    val_inds = indices[int(train_ratio*len(subject_id)):int((train_ratio + val_ratio)*len(subject_id))]
    test_inds = indices[int((train_ratio + val_ratio)*len(subject_id)):]
    
    def split(x):
        if x in subject_id[train_inds]:
            return 'train'
        elif x in subject_id[val_inds]:
            return 'val'
        elif x in subject_id[test_inds]:
            return 'test'
        
    split_dict = dict(zip(subject_id, list(map(split, subject_id))))
    
    try:
        cnter = Counter(table_df['ICU Admit'].values[train_inds])
        assert cnter['yes'] > 1
        cnter = Counter(table_df['ICU Admit'].values[val_inds])
        assert cnter['yes'] > 1
        cnter = Counter(table_df['ICU Admit'].values[test_inds])
        assert cnter['yes'] > 1
    except AssertionError:
        print('Try some other random seeds.')
    
    # subject_id | image_dir | image_id |     

    png_dirs = list(filter(lambda x: PurePath(x).parts[-4] in subject_id, png_dirs))
    labels = []
    image_ids = []
    subject_ids = []
    splits = []
    value_dict = {}
    
    for c in CATEGORICAL_IDS + CONTINOUS_IDS:
        value_dict[c] = []
    
    for png in tqdm(png_dirs):
        image_id = '$$'.join(PurePath(png).parts[-4:])[:-4]
        image_ids.append(image_id)
        
        subject_id_ = PurePath(png).parts[-4] 
        subject_ids.append(subject_id_)
        
        split_ = split_dict[subject_id_]
        splits.append(split_)
        
        label = table_df[table_df['PATIENT_ID']==subject_id_]['ICU Admit'].values[0]
        label = 1 if label == 'yes' else 0
        labels.append(label)
        
        for c in CATEGORICAL_IDS + CONTINOUS_IDS:
            value_dict[c].append(table_df[table_df['PATIENT_ID']==subject_id_][c].values[0])
        
    value_dict['SUBJECT ID'] = subject_ids
    value_dict['SPLIT'] = splits
    value_dict['LABEL'] = labels
    value_dict['IMAGE ID'] = image_ids
    value_dict['IMAGE DIRS'] = png_dirs
    
    for k, v in zip(value_dict.keys(), value_dict.values()):
        print(k, len(v))
    
    value_dict = pd.DataFrame(value_dict)
    value_dict.to_csv(join(ROOT_PATH, 'data.csv'), index=False)

if __name__ == "__main__":
    meta_df, table_df = read_csv(META_PATH, TABLE_PATH)
    dcm_dirs = collect_images(IMAGE_PATH, suffix='dcm')
    png_dirs = collect_images(IMAGE_PATH, suffix='png')
    if len(png_dirs) != len(dcm_dirs):
        print('Convert DCM to PNG')
        iterate_on_images(dcm_dirs) # convert dcm images to png images
    # analyse_columns(table_df)
    # process_columns(table_df)
    processed_table_df = process_columns(table_df)
    # analyse_columns(processed_table_df)
    wrapup(IMAGE_PATH, processed_table_df, png_dirs)