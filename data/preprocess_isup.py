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
from functools import partial

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

def get_area_slice(mask_slice, id_):
    # mask (b,w,h)
    # mask_slice (b, w)
    # label (0, 1, 2, 3, 4)
    area = (mask_slice == id_).sum()
    return area

def get_area(masks, id_):
    # masks: List[(b, w, h)]
    area = list(map(partial(get_area_slice, id_), masks))
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
    lesion_ids = list(set(np.array(mask, dtype=np.int64).flatten()))
    lesion_ids = list(sorted(lesion_ids, reverse=True))
    lesion_ids = list(filter(lambda x: x!=0, lesion_ids))
    lesion_ids = list(filter(lambda x: x!=1, lesion_ids))
    combined_img = [combined_img[...,i,:] for i in range(combined_img.shape[-2])]
    mask = [mask[...,i] for i in range(mask.shape[-1])]
    res_mask = []
    res_image = []
    selected_img = {}
    for id_ in lesion_ids:
        selected_img[id_] = []
    
    for id_ in lesion_ids:
        area = get_area(mask, id_)
        # filter slices with at least 5 label 2
        # thres = 10
        # slices = list(filter(lambda x: area[x] >= thres, area.keys()))
        slices = list(sorted(area.keys(), key=lambda x: area[x]))
        thres = 10
        slices = list(filter(lambda x: area[x] >= thres, slices))        
        
        for s in range(len(mask)):
            if s in slices:
                selected_img[id_].append(combined_img[s])
            else:
                res_mask.append(mask[s])
                res_image.append(combined_img[s])
        mask = res_mask
        combined_img = res_image
        res_mask = []
        res_image = []
        
    return selected_img

def generate_image():
    folders = glob(join(ROOT, 'processed_data', '*'))
    for f in tqdm(folders):
        img, mask = read_folder(f)
        selected_img = crop_roi(img, mask)
        for id_ in selected_img.keys():
            for i, s in enumerate(selected_img[id_], 0):
                s = np.array(s)
                np.save(join(f, f'IMAGE_{id_-1}_{i}.npy'), s)     

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
    img_dirs = glob(join(ROOT, 'processed_data', '*', 'IMAGE*.npy'))
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
        # lesion_id = subject_id + '_1'
        id_ = int(PurePath(img_dir).parts[-1].split('_')[1])
        lesion_id = subject_id + f'_{id_}'
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
    
    # train_subject = []
    # val_subject = []
    # test_subject = []
    # # there are three labels, {0, 1, 2}
    # # for label==0
    # subject_id = np.array(list(set(df[df['label']==0]['SubjectID'].values))) 
    # indices = list(range(len(subject_id)))
    # np.random.shuffle(indices)
    # train_inds_ = indices[:int(train_ratio*len(subject_id))]
    # test_inds_ = indices[int(train_ratio*len(subject_id)):int((train_ratio + test_ratio)*len(subject_id))]
    # val_inds_ = indices[int((train_ratio + test_ratio)*len(subject_id)):]
    # train_subject = list(subject_id[train_inds_]) + train_subject
    # val_subject = list(subject_id[val_inds_]) + val_subject
    # test_subject = list(subject_id[test_inds_]) + test_subject

    # # for label==1
    # subject_id = np.array(list(set(df[df['label']==1]['SubjectID'].values))) 
    # indices = list(range(len(subject_id)))
    # np.random.shuffle(indices)
    # train_inds_ = indices[:int(train_ratio*len(subject_id))]
    # test_inds_ = indices[int(train_ratio*len(subject_id)):int((train_ratio + test_ratio)*len(subject_id))]
    # val_inds_ = indices[int((train_ratio + test_ratio)*len(subject_id)):]
    # train_subject = list(subject_id[train_inds_]) + train_subject
    # val_subject = list(subject_id[val_inds_]) + val_subject
    # test_subject = list(subject_id[test_inds_]) + test_subject
    
    # # for label==2
    # subject_id = np.array(list(set(df[df['label']==2]['SubjectID'].values))) 
    # indices = list(range(len(subject_id)))
    # np.random.shuffle(indices)
    # train_inds_ = indices[:int(train_ratio*len(subject_id))]
    # test_inds_ = indices[int(train_ratio*len(subject_id)):int((train_ratio + test_ratio)*len(subject_id))]
    # val_inds_ = indices[int((train_ratio + test_ratio)*len(subject_id)):]
    # train_subject = list(subject_id[train_inds_]) + train_subject
    # val_subject = list(subject_id[val_inds_]) + val_subject
    # test_subject = list(subject_id[test_inds_]) + test_subject
    
    train_subject = ['PAIs088', 'PAIs155', 'PAIs137', 'PAIs062', 'PAIs198', 'PAIs197', 'PAIs105', 'PAIs037', 'PAIs036', 'PAIs015', 'PAIs163', 'PAIs120', 'PAIs066', 'PAIs109', 'PAIs058', 'PAIs003', 'PAIs194', 'PAIs095', 'PAIs237', 'PAIs218', 'PAIs042', 'PAIs230', 'PAIs229', 'PAIs239', 'PAIs192', 'PAIs183', 'PAIs231', 'PAIs113', 'PAIs168', 'PAIs226', 'PAIs150', 'PAIs047', 'PAIs020', 'PAIs243', 'PAIs084', 'PAIs209', 'PAIs177', 'PAIs070', 'PAIs002', 'PAIs221', 'PAIs054', 'PAIs078', 'PAIs025', 'PAIs077', 'PAIs202', 'PAIs242', 'PAIs039', 'PAIs188', 'PAIs206', 'PAIs141', 'PAIs135', 'PAIs189', 'PAIs140', 'PAIs125', 'PAIs012', 'PAIs184', 'PAIs191', 'PAIs063', 'PAIs080', 'PAIs075', 'PAIs139', 'PAIs101', 'PAIs061', 'PAIs208', 'PAIs114', 'PAIs187', 'PAIs228', 'PAIs116', 'PAIs123', 'PAIs064', 'PAIs145', 'PAIs014', 'PAIs224', 'PAIs072', 'PAIs013', 'PAIs149', 'PAIs083', 'PAIs074', 'PAIs124', 'PAIs005', 'PAIs166', 'PAIs179', 'PAIs016', 'PAIs096', 'PAIs151', 'PAIs121', 'PAIs028', 'PAIs157', 'PAIs052', 'PAIs159', 'PAIs240', 'PAIs048', 'PAIs007', 'PAIs085', 'PAIs073', 'PAIs051', 'PAIs153', 'PAIs112', 'PAIs093', 'PAIs108', 'PAIs147', 'PAIs204', 'PAIs217', 'PAIs185', 'PAIs033', 'PAIs098', 'PAIs090', 'PAIs067', 'PAIs180', 'PAIs094', 'PAIs173', 'PAIs010', 'PAIs154', 'PAIs215', 'PAIs110', 'PAIs017', 'PAIs238', 'PAIs174', 'PAIs142', 'PAIs035', 'PAIs162', 'PAIs213', 'PAIs055']
    val_subject = ['PAIs210', 'PAIs212', 'PAIs006', 'PAIs170', 'PAIs057', 'PAIs022', 'PAIs103', 'PAIs225', 'PAIs118', 'PAIs169', 'PAIs146', 'PAIs068', 'PAIs175', 'PAIs060', 'PAIs129', 'PAIs236', 'PAIs021', 'PAIs195', 'PAIs069', 'PAIs126', 'PAIs071', 'PAIs167', 'PAIs099', 'PAIs176', 'PAIs127', 'PAIs164', 'PAIs027', 'PAIs128', 'PAIs031', 'PAIs018', 'PAIs019', 'PAIs026', 'PAIs241', 'PAIs024', 'PAIs030', 'PAIs223', 'PAIs148', 'PAIs138', 'PAIs152', 'PAIs220', 'PAIs029', 'PAIs050']
    test_subject = ['PAIs160', 'PAIs107', 'PAIs046', 'PAIs011', 'PAIs232', 'PAIs065', 'PAIs219', 'PAIs199', 'PAIs165', 'PAIs001', 'PAIs004', 'PAIs053', 'PAIs214', 'PAIs038', 'PAIs156', 'PAIs076', 'PAIs043', 'PAIs045', 'PAIs201', 'PAIs130', 'PAIs082', 'PAIs161', 'PAIs040', 'PAIs056', 'PAIs222', 'PAIs171', 'PAIs136', 'PAIs117', 'PAIs131', 'PAIs143', 'PAIs091', 'PAIs115', 'PAIs059', 'PAIs132', 'PAIs086', 'PAIs104', 'PAIs182', 'PAIs044', 'PAIs009', 'PAIs158', 'PAIs032']

    # subject_id = np.array(subject_id)
    
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
    split_data()