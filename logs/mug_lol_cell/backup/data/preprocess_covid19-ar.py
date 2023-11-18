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

def add_jitter_float(x, jitter=10):
    return x + (round(np.random.rand(), 2)-0.5)*2*jitter
    
# wrapup all information in a table
def wrapup(table_df, png_dirs):
    subject_id = table_df['PATIENT_ID'].values 

    png_dirs = list(filter(lambda x: PurePath(x).parts[-4] in subject_id, png_dirs))
    labels = []
    image_ids = []
    subject_ids = []
    value_dict = {}
    
    for c in CATEGORICAL_IDS + CONTINOUS_IDS:
        value_dict[c] = []
    
    for png in tqdm(png_dirs):
        image_id = '$$'.join(PurePath(png).parts[-4:])[:-4]
        image_ids.append(image_id)
        
        subject_id_ = PurePath(png).parts[-4] 
        subject_ids.append(subject_id_)
        
        label = table_df[table_df['PATIENT_ID']==subject_id_]['ICU Admit'].values[0]
        label = 1 if label == 'yes' else 0
        labels.append(label)
        
        for c in CATEGORICAL_IDS + CONTINOUS_IDS:
            value_dict[c].append(table_df[table_df['PATIENT_ID']==subject_id_][c].values[0])
        
    value_dict['SUBJECT ID'] = subject_ids
    value_dict['LABEL'] = labels
    value_dict['IMAGE ID'] = image_ids
    value_dict['IMAGE DIRS'] = png_dirs
    
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
    df['AGE'] = df['AGE'].apply(add_jitter, args=(3,))
    df['AGE'] = df['AGE'].astype('int')
    df['LATEST WEIGHT'] = df['LATEST WEIGHT'].apply(add_jitter_float, args=(3,))
    df['LATEST HEIGHT'] = df['LATEST HEIGHT'].apply(add_jitter_float, args=(3,))
    df['LATEST_BMI'] = (df['LATEST WEIGHT'].values*0.453592) / ((df['LATEST HEIGHT'].values/100)**2)
    df['LATEST_BMI'] = df['LATEST_BMI'].round(2)
    df['LATEST HEIGHT'] = df['LATEST HEIGHT'].round(2)
    df['LATEST WEIGHT'] = df['LATEST WEIGHT'].round(2)
        
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
    test_ratio = 0.2 # val ratio = 0.2
    
    # for label == 0
    subject_id = np.array(list(set(df[df['LABEL']==0]['SUBJECT ID'].values))) 
    indices = list(range(len(subject_id)))
    np.random.shuffle(indices)
    train_inds_ = indices[:int(train_ratio*len(subject_id))]
    test_inds_ = indices[int(train_ratio*len(subject_id)):int((train_ratio + test_ratio)*len(subject_id))]
    val_inds_ = indices[int((train_ratio + test_ratio)*len(subject_id)):]
    train_subject = list(subject_id[train_inds_])
    val_subject = list(subject_id[val_inds_])
    test_subject = list(subject_id[test_inds_])
    
    
    # for label == 1
    subject_id = np.array(list(set(df[df['LABEL']==1]['SUBJECT ID'].values))) 
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
        
    splits = np.array(list(map(split, df['SUBJECT ID'])))
    df['SPLIT'] = splits
    
    
    df.to_csv(join(ROOT_PATH, 'data_full.csv'), index=False)
    # with open(join(ROOT_PATH, 'meta.json'), 'w') as f:
    #     f.write(json.dumps(meta, indent=4))
    
if __name__ == "__main__":
    meta_df, table_df = read_csv(META_PATH, TABLE_PATH)
    dcm_dirs = collect_images(IMAGE_PATH, suffix='dcm')
    png_dirs = collect_images(IMAGE_PATH, suffix='png')
    if len(png_dirs) != len(dcm_dirs):
        print('Convert DCM to PNG')
        iterate_on_images(dcm_dirs) # convert dcm images to png images
    # analyse_columns(table_df)
    processed_table_df = process_columns(table_df)
    # analyse_columns(processed_table_df)
    # wrapup(processed_table_df, png_dirs)
    postprocess()
    