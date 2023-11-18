import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
import nibabel as nib
from collections import Counter
from PIL import Image
from os.path import join
import os
import json
from pathlib import PurePath

ROOT = 'data/prostate_ISUP/'

def read_nii(img_dir):
    img = nib.load(img_dir).get_fdata()
    return img

def preprocess(img):
    # normalize
    img = (img - img.min()) / (img.max() - img.min())
    # expand dims
    img = np.expand_dims(img, axis=-1)
    return img

def get_area_slice(mask_slice):
    # mask (b,w,h)
    # mask_slice (b, w)
    # label (0, 1, 2)
    # area_1 = (mask_slice == 1).sum()
    area_2 = (mask_slice == 2).sum()
    # return (area_1, area_2)
    return area_2

def get_area(mask):
    # mask (b, w, h)
    mask = [mask[...,i] for i in range(mask.shape[-1])]
    area = list(map(get_area_slice, mask))
    return dict(zip(list(range(len(area))), area))

def read_folder(folder):
    # there are three modalities: ADC, DWI, and T2W
    adc = read_nii(join(folder, 'ADC.nii.gz'))
    dwi = read_nii(join(folder, 'DWI.nii.gz'))
    t2w = read_nii(join(folder, 'T2W.nii.gz'))
    
    adc = preprocess(adc)
    dwi = preprocess(dwi)
    t2w = preprocess(t2w)
    
    combined_img = np.concatenate((adc, dwi, t2w), axis=-1)
    
    mask = read_nii(join(folder, 'Con_gt.nii.gz'))
    
    return combined_img, mask   

def crop_roi(combined_img, mask):
    # combined_img (b, w, h, 3), mask (b, w, h)
    area = get_area(mask)
    # filter slices with at least 5 label 2
    thres = 10
    slices = list(filter(lambda x: area[x] >= thres, area.keys()))
    selected_img = [combined_img[...,s,:] for s in slices]
    return selected_img

def generate_image():
    folders = glob(join(ROOT, 'processed_data', '*'))
    for f in tqdm(folders):
        img, mask = read_folder(f)
        selected_img = crop_roi(img, mask)
        for i, s in enumerate(selected_img, 0):
            s = np.array(s)
            np.save(join(f, f'img_{i}.npy'), s)    


def read_folder1(folder):
    # there are three modalities: ADC, DWI, and T2W
    t2w = read_nii(join(folder, 'T2W.nii.gz'))
    
    t2w = preprocess(t2w)
    
    combined_img = np.concatenate((t2w, t2w, t2w), axis=-1)
    
    mask = read_nii(join(folder, 'Con_gt.nii.gz'))
    
    return combined_img, mask   

def generate_image1():
    folders = glob(join(ROOT, 'processed_data', '*'))
    for f in tqdm(folders):
        img, mask = read_folder1(f)
        selected_img = crop_roi(img, mask)
        for i, s in enumerate(selected_img, 0):
            s = np.array(s)
            np.save(join(f, f'img_{i}1.npy'), s)    

def map_label(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return 2

def check_columns():
    info_csv = pd.read_csv(join(ROOT, 'PAIs_Details_and_Labels_send2AIteam_updated20231001.csv'))
    columns = ['Age', 'Weight', 'DRE_2y1u0n', 'USGlesion_2y1u0n', 
            'Volume_gt', 'Volume_TRUS', 'Volume_MRI', 'LesionsNumber_PerPatient_gt'] + ['PSA_Latest', 'PSAD_PSA_Latest/Volume_gt',
            'PHI_Latest', 'PHID_PHI_Latest/Volume_gt', 'Creatinine_conc_ppm', 'CreD_Creatinine/Volume_gt', 'Normalised_Put',
            'PutD_Put/Volume_gt', 'Normalised_Spd', 'SpdD_Spd/Volume_gt', 'Normalised_Spm', 'SpmD_Spm/Volume_gt', 'Analysis_Location'] # a collection of clinical features and biomarkers
    drop_columns = []
    for c in columns:
        if len(info_csv[info_csv[c]=='/']) >= 0.2*len(info_csv): # only keep columns with at least 80% coverage on the data
            drop_columns.append(c)  
    print(drop_columns)
    columns = list(filter(lambda x: x not in drop_columns, columns))
    print(columns)

def fill_column(column, label, num=1):
    # pick some samples from the same data distribution and take the average
    # for categorical columns, 'num' must stay 1
    label = np.array(label)
    for i in range(len(column)):
        if column.values[i] == '/':
            l = label[i]
            index = column.values!= '/'
            label_ = label[index]
            values = column.values[index]
            dist = values[label_==l]
            filled_value = [float(np.random.choice(dist)) for n in range(num)]
            filled_value = np.mean(filled_value)
            column.iloc[i] = filled_value
    return column

def write_csv():
    info_csv = pd.read_csv(join(ROOT, 'PAIs_Details_and_Labels_send2AIteam_updated20231001.csv'))
    
    label = info_csv['Lesions_ISUP_max'].values
    label = list(map(map_label, label))
    print(Counter(label))
    
    js = {}
    
    continuous_columns = ['Age', 'Weight', 'Volume_gt', 'Volume_TRUS', 'Volume_MRI', 
                            'PSA_Latest', 'PSAD_PSA_Latest/Volume_gt', 'Creatinine_conc_ppm',
                            'CreD_Creatinine/Volume_gt', 'Normalised_Spm', 'SpmD_Spm/Volume_gt']
    categorical_columns = ['DRE_2y1u0n', 'USGlesion_2y1u0n', 'LesionsNumber_PerPatient_gt']
    
    columns = continuous_columns + categorical_columns
    
    df = info_csv.loc[:, columns]
    
    for c in categorical_columns:
        fill_column(df[c], label, num=1)
        df[c] = df[c].astype('int32')
        js[c] = {
        "field_length": len(set(df[c].values)),
        "type": "categorical",
        "full_name": ""
        }
    
    for c in continuous_columns:
        fill_column(df[c], label, num=10)
        df[c] = df[c].astype('float32')
        js[c] = {
        "field_length": 1,
        "type": "continuous",
        "full_name": ""
        }
    
    df['label'] = label
    df.to_csv(join(ROOT, 'data.csv'), index=False)
    
    with open(join(ROOT, 'meta.json'), 'w') as f:
        f.write(json.dumps(js, indent=4))  
    
def complete_csv():
    img_dirs = glob(join(ROOT, 'processed_data', '*', '*.npy'))
    info_csv = pd.read_csv(join(ROOT, 'PAIs_Details_and_Labels_send2AIteam_updated20231001.csv'))
    
    df = pd.read_csv(join(ROOT, 'data.csv'))
    df['Prostate_AI_CaseID'] = info_csv['Prostate_AI_CaseID'] 
    df['Prostate_AI_LesionID'] = info_csv['Prostate_AI_LesionID'] 
    continuous_columns = ['Age', 'Weight', 'Volume_gt', 'Volume_TRUS', 'Volume_MRI', 
                            'PSA_Latest', 'PSAD_PSA_Latest/Volume_gt', 'Creatinine_conc_ppm',
                            'CreD_Creatinine/Volume_gt', 'Normalised_Spm', 'SpmD_Spm/Volume_gt']
    categorical_columns = ['DRE_2y1u0n', 'USGlesion_2y1u0n', 'LesionsNumber_PerPatient_gt']
    columns = continuous_columns + categorical_columns + ['label']
    column_values = {}
    for c in columns:
        column_values[c] = []
    # column_values = dict(zip(columns, []*len(columns)))
    
    df_ = {}
    
    subject_ids = []
    for img_dir in tqdm(img_dirs):
        subject_id = PurePath(img_dir).parts[-2]
        lesion_id = subject_id + '_1'
        subject_ids.append(subject_id)
        for c in columns:
            column_values[c].append(df[df['Prostate_AI_LesionID']==lesion_id][c].values[0])
    
    df_['SubjectID'] = np.array(subject_ids)
    df_['Image_DIR'] = np.array(img_dirs)
    for c in columns:
        df_[c] = np.array(column_values[c])
    
    df_ = pd.DataFrame(df_)
    
    for c in continuous_columns:
        v = df_[c].values
        df_[c+'_num'] = (v - v.mean()) / v.std()
        
    for c in categorical_columns:
        v = df_[c].values 
        uv = list(set(v))
        mapping_dict = dict(zip(uv, list(range(len(uv)))))
        v = list(map(lambda x: mapping_dict[x], v))
        df_[c + '_num'] = np.array(v)
    
    df_.to_csv(join(ROOT, 'data_processed.csv'), index=False)

def split_data():
    df = pd.read_csv(join(ROOT, 'data_processed.csv'))
    
    # train, val, test 0.6, 0.2, 0.2
    train_ratio = 0.6
    test_ratio = 0.2 # val_ratio = 0.2
    
    train_subject = []
    val_subject = []
    test_subject = []
    # there are three labels, {0, 1, 2}
    # for label==0
    subject_id = np.array(list(set(df[df['label']==0]['SubjectID'].values))) 
    indices = list(range(len(subject_id)))
    np.random.shuffle(indices)
    train_inds_ = indices[:int(train_ratio*len(subject_id))]
    test_inds_ = indices[int(train_ratio*len(subject_id)):int((train_ratio + test_ratio)*len(subject_id))]
    val_inds_ = indices[int((train_ratio + test_ratio)*len(subject_id)):]
    train_subject = list(subject_id[train_inds_]) + train_subject
    val_subject = list(subject_id[val_inds_]) + val_subject
    test_subject = list(subject_id[test_inds_]) + test_subject

    # for label==1
    subject_id = np.array(list(set(df[df['label']==1]['SubjectID'].values))) 
    indices = list(range(len(subject_id)))
    np.random.shuffle(indices)
    train_inds_ = indices[:int(train_ratio*len(subject_id))]
    test_inds_ = indices[int(train_ratio*len(subject_id)):int((train_ratio + test_ratio)*len(subject_id))]
    val_inds_ = indices[int((train_ratio + test_ratio)*len(subject_id)):]
    train_subject = list(subject_id[train_inds_]) + train_subject
    val_subject = list(subject_id[val_inds_]) + val_subject
    test_subject = list(subject_id[test_inds_]) + test_subject
    
    # for label==2
    subject_id = np.array(list(set(df[df['label']==2]['SubjectID'].values))) 
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
        
    splits = np.array(list(map(split, df['SubjectID'])))
    
    df['split'] = splits
    
    print(Counter(df[df['split']=='train']['label'].values))
    print(Counter(df[df['split']=='val']['label'].values))
    print(Counter(df[df['split']=='test']['label'].values))
    
    df.to_csv(join(ROOT, 'data_full.csv'), index=False)
    

if __name__ == "__main__": 
    # generate_image()
    # check_columns()
    # write_csv()
    # complete_csv()
    # split_data()
    generate_image1()