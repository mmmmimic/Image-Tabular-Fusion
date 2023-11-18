'''
Preprocess DVM data and store tabular information in csv files
Modified from https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/data/create_dvm_dataset.ipynb 
'''
import pandas as pd
import os
from os.path import join
import torch
import random
import numpy as np
import json

#----------------------------------------------------------------------
BASE = 'data/dvm_car'
TABLES = join(BASE, 'dvm/19586296/tables_V2.0')
FEATURES = join(BASE, 'features')
ANALYSIS = join(BASE, 'analysis')
#----------------------------------------------------------------------

front_view_only = False # random sampling images from different views
categorical_ids = ['Color',
'Bodytype',
'Gearbox',
'Fuel_type',
'Genmodel_ID']
continuous_ids = [
    'Adv_year',
    'Adv_month',
    'Reg_year',
    'Runned_Miles',
    'Price',
    'Seat_num',
    'Door_num',
    'Entry_price', 
    'Engine_size']

def conf_matrix_from_matrices(mat_gt, mat_pred):
  overlap_and = (mat_pred & mat_gt)
  tp = overlap_and.sum()
  fp = mat_pred.sum()-overlap_and.sum()
  fn = mat_gt.sum()-overlap_and.sum()
  tn = mat_gt.shape[0]**2-(tp+fp+fn)
  return tp, fp, fn, tn

def check_or_save(obj, path, index=None, header=None):
  if isinstance(obj, pd.DataFrame):
    if index is None or header is None:
      raise ValueError('Index and header must be specified for saving a dataframe')
    if os.path.exists(path):
      if not header:
        saved_df = pd.read_csv(path,header=None)
      else:
        saved_df = pd.read_csv(path)
      naked_df = saved_df.reset_index(drop=True)
      naked_df.columns = range(naked_df.shape[1])
      naked_obj = obj.reset_index(drop=not index)
      naked_obj.columns = range(naked_obj.shape[1])
      if naked_df.round(6).equals(naked_obj.round(6)):
        return
      else:
        diff = (naked_df.round(6) == naked_obj.round(6))
        diff[naked_df.isnull()] = naked_df.isnull() & naked_obj.isnull()
        assert diff.all().all(), "Dataframe is not the same as saved dataframe"
    else:
      obj.to_csv(path, index=index, header=header)
  else:
    if os.path.exists(path):
      saved_obj = torch.load(path)
      if isinstance(obj, list):
        for i in range(len(obj)):
          check_array_equality(obj[i], saved_obj[i])
      else:
        check_array_equality(obj, saved_obj)
    else:
      print(f'Saving to {path}')
      torch.save(obj, path)

def check_array_equality(ob1, ob2):
  if torch.is_tensor(ob1) or isinstance(ob1, np.ndarray):
    assert (ob2 == ob1).all()
  else:
    assert ob2 == ob1

def read_csv(csv_dir):
    data = pd.read_csv(csv_dir, low_memory=False)
    col_names = data.columns.values
    for name in col_names:
        # remove the colums start with space
        if name.startswith(' '):
            data[name[1:]] = data[name].values
            data = data.drop(name, axis=1)
    return data

def parser_adv_id(x):
  split = x["Image_ID"].split('$$')
  return f"{split[0]}$${split[1]}"

def extract_engine_size(x):
  return float(x['Engin_size'][:-1])

def get_ids(split):
# for train, val and test ids, keep them the same with the paper
  with open(join(BASE,f'{split}_ids_dvm_all_views.txt'), 'r') as f:
    c = f.read()
  return np.array(c.split(',')[:-1]) # last element is a space

def get_paths(df):
    paths = []
    for _, row in df.iterrows():
        im_name = row['Image_name']
        split = im_name.split('$$')
        path = join(BASE, 'dvm/19586296/resized_DVM_v2/resized_DVM/', split[0], split[1], split[2], split[3], im_name)
        paths.append(path)
    return paths

def add_jitter(x, jitter=50):
    return x + random.randint(-jitter, jitter)

def fill_from_other_entry(row):
    for attr in ['Wheelbase', 'Length', 'Width', 'Height']:
        if pd.isna(row[attr]) or row[attr]==0:
            other_rows = physical_df_orig.loc[physical_df_orig['Genmodel_ID']==row['Genmodel_ID']]
            other_rows.dropna(subset=[attr], inplace=True)
            other_rows.drop_duplicates(subset=[attr], inplace=True)
            other_rows = other_rows[other_rows[attr]>0]
            if len(other_rows)>0:
                row[attr] = other_rows[attr].values[0]
    return row

if __name__ == "__main__":
    # read tables
    ad_data = read_csv(join(TABLES, 'Ad_table.csv'))
    basic_data = read_csv(join(TABLES, 'Basic_table.csv'))
    image_data = read_csv(join(TABLES, 'Image_table.csv'))
    price_data = read_csv(join(TABLES, 'Price_table.csv'))
    sales_data = read_csv(join(TABLES, 'Sales_table.csv'))
    trim_data = read_csv(join(TABLES, 'Trim_table.csv'))
    
    # parse image id and remove duplicates
    image_data["Adv_ID"] = image_data.apply(lambda x: parser_adv_id(x), axis=1)
    if front_view_only:
        image_data = image_data[(image_data["Quality_check"]=="P")&(image_data["Predicted_viewpoint"]==0)]
    image_data.drop_duplicates(subset=['Adv_ID'], inplace=True)
    
    feature_df = ad_data.merge(price_data[['Genmodel_ID', 'Entry_price', 'Year']], left_on=['Genmodel_ID','Reg_year'], right_on=['Genmodel_ID','Year'])
        
    data_df = feature_df.merge(image_data[['Adv_ID', 'Image_name', 'Predicted_viewpoint']], left_on=['Adv_ID'], right_on=['Adv_ID'])
    assert data_df["Adv_ID"].is_unique
    
    # convert engine size to numbers
    data_df.dropna(inplace=True)
    data_df['Engine_size'] = data_df.apply(lambda x: extract_engine_size(x), axis=1)
    data_df.drop(columns=['Engin_size'], inplace=True)
    
    id_df = data_df.loc[:,'Adv_ID']
    image_name_df = data_df.loc[:,'Image_name']
    viewpoint_df = data_df.loc[:,'Predicted_viewpoint']
    continuous_df = data_df.loc[:,continuous_ids]
    categorical_df = data_df.loc[:,categorical_ids]
    continuous_df['Runned_Miles'] = pd.to_numeric(continuous_df['Runned_Miles'], errors='coerce')
    continuous_df['Price'] = pd.to_numeric(continuous_df['Price'], errors='coerce')

    # normalize
    continuous_df_ = continuous_df.copy()
    for c in ['Adv_year', 'Adv_month', 'Reg_year', 'Runned_Miles', 'Price', 'Seat_num', 'Door_num', 'Entry_price', 'Engine_size']:
      continuous_df_[c + '_num'] = continuous_df_[c].values
      continuous_df_.drop([c], axis=1, inplace=True)
    continuous_df_ = (continuous_df_-continuous_df_.mean())/continuous_df_.std()
    continuous_df = pd.concat([continuous_df, continuous_df_], axis=1)
    
    # convert labels to Integer ID
    categorical_df['Genmodel_ID'] = categorical_df['Genmodel_ID'].astype('category')
    categorical_df['Color_num'] = categorical_df['Color'].astype('category')
    categorical_df['Bodytype_num'] = categorical_df['Bodytype'].astype('category')
    categorical_df['Gearbox_num'] = categorical_df['Gearbox'].astype('category')
    categorical_df['Fuel_type_num'] = categorical_df['Fuel_type'].astype('category')

    cat_columns = categorical_df.select_dtypes(['category']).columns

    categorical_df[cat_columns] = categorical_df[cat_columns].apply(lambda x: x.cat.codes)

    data_df = pd.concat([id_df, continuous_df, categorical_df, image_name_df, viewpoint_df], axis=1)
    data_df.dropna(inplace=True)
    
    minimum_population = 100
    values = (data_df.value_counts(subset=['Genmodel_ID'])>=minimum_population).values
    codes = (data_df.value_counts(subset=['Genmodel_ID'])>=minimum_population).index
    populated_codes = []
    for i, v in enumerate(values):
        if v:
            populated_codes.append(int(codes[i][0]))
    
    data_df = data_df[data_df['Genmodel_ID'].isin(populated_codes)]
    map = {}
    for i,l in enumerate(data_df['Genmodel_ID'].unique()):
        map[l] = i
    data_df['Genmodel_ID'] = data_df['Genmodel_ID'].map(map)
    
    data_df.to_csv(join(BASE, 'data.csv'), index=False)
    
    # # detect bad indices
    # bad_indices = []
    # for indx, row in data_df.iterrows():
    #     im_name = row['Image_name']
    #     split = im_name.split('$$')
    #     path = join(BASE, 'dvm/19586296/resized_DVM_v2/resized_DVM/', split[0], split[1], split[2], split[3], im_name)
    #     if not os.path.exists(path):
    #         bad_indices.append(indx)

    _ids = list(data_df['Adv_ID'])
    addendum = '_all_views'
    non_feature_columns = ['Adv_ID', 'Image_name', 'Predicted_viewpoint', 'Genmodel_ID']

    train_ids = get_ids('train')
    val_ids = get_ids('val')
    test_ids = get_ids('test')

    train_df = data_df.set_index('Adv_ID').loc[train_ids]
    val_df = data_df.set_index('Adv_ID').loc[val_ids]
    test_df = data_df.set_index('Adv_ID').loc[test_ids]

    train_labels_all = list(train_df['Genmodel_ID'])
    val_labels_all = list(val_df['Genmodel_ID'])
    test_labels_all = list(test_df['Genmodel_ID'])

    train_df.loc[:,~train_df.columns.isin(non_feature_columns)].to_csv(join(BASE, 'train_df.csv'))
    val_df.loc[:,~val_df.columns.isin(non_feature_columns)].to_csv(join(BASE, 'val_df.csv'))
    test_df.loc[:,~test_df.columns.isin(non_feature_columns)].to_csv(join(BASE, 'test_df.csv'))

    np.save(join(BASE, 'train_labels.npy'), train_labels_all)
    np.save(join(BASE, 'val_labels.npy'), val_labels_all)
    np.save(join(BASE, 'test_labels.npy'), test_labels_all)
    

    # save image paths
    for df, name in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
        paths = get_paths(df)
        paths = np.array(paths)
        np.save(join(BASE, f'{name}_paths.npy'), paths)
    
    # adding missing values to physical table
    # Fill using other values
    physical_df_orig = read_csv(join(BASE, 'dvm/19586296/tables_V2.0/Ad_table_physical.csv'))
    
    # Manual touches
    
    # Peugeot RCZ
    physical_df_orig.loc[physical_df_orig['Genmodel_ID'] == '69_36','Wheelbase']=2612
    # Ford Grand C-Max
    physical_df_orig.loc[physical_df_orig['Genmodel_ID'] == '29_20','Wheelbase']=2788 
    
    physical_df_orig = physical_df_orig.apply(fill_from_other_entry, axis=1)

    physical_df_orig.to_csv(join(BASE, 'Ad_table_physical_filled.csv'), index=False)
    
    # jitter 50
    random.seed(2022)
    physical_df = physical_df_orig.copy()
    for attr in ['Wheelbase', 'Length', 'Width', 'Height']:
        physical_df[attr] = physical_df[attr].apply(add_jitter)
    physical_df.to_csv(join(BASE, 'Ad_table_physical_filled_jittered_50.csv'), index=False)
    
    # # Ford ranger (29_30) has wrong height. Missing 1 in front... 805.0 instead of 1805.0
    # # Mercedes Benz (59_29) wrong wheelbase, 5246.0 instead of 3106
    # # Kia Rio (43_9) wrong wheelbase, 4065.0 instead of 2580
    # # FIXED
    
    physical_df = pd.read_csv(join(BASE,'Ad_table_physical_filled_jittered_50.csv'))[['Adv_ID', 'Wheelbase','Height','Width','Length']]
    for split in ['train', 'val', 'test']:
        features_df = pd.read_csv(join(BASE, f'{split}_df.csv'))
        merged_df = features_df.merge(physical_df, on='Adv_ID')

        for attr in ['Wheelbase','Height','Width','Length']:
            assert merged_df[attr].isna().sum()==0
            assert (merged_df[attr]==0).sum()==0

        # normalize physical attributes
        for attr in ['Wheelbase','Height','Width','Length']:
            merged_df[attr+'_num'] = (merged_df[attr]-merged_df[attr].mean())/merged_df[attr].std()

        merged_df['Bodytype'] = merged_df['Bodytype'].astype('category')

        merged_df[['Bodytype_num']] = merged_df[['Bodytype']].apply(lambda x: x.cat.codes)

        # Drop unwanted cols
        # non_feature_columns = ['Adv_ID', 'Image_name', 'Genmodel_ID']
        non_feature_columns = ['Adv_ID']
        merged_df = merged_df.drop(non_feature_columns, axis=1)

        merged_df_cols = merged_df.columns.tolist()
        rearranged_cols = merged_df_cols[-4:]+merged_df_cols[:-4]
        merged_df = merged_df[rearranged_cols]
        
        merged_df.to_csv(join(BASE, f'{split}_df_full.csv'), index=False)
    # save meta information of each column, including field length, type ['categorical', 'continuous', 'non-feature']
    physical_train = pd.read_csv(join(BASE, 'train_df_full.csv'))
    physical_val = pd.read_csv(join(BASE, 'val_df_full.csv'))
    physical_test = pd.read_csv(join(BASE, 'test_df_full.csv'))
    data_df = pd.concat([physical_train, physical_val, physical_test], axis=0)
    meta = {}
    continuous_ids += ['Wheelbase','Height','Width','Length']
    categorical_ids.pop(categorical_ids.index('Genmodel_ID'))
    categorical_ids += ['Bodytype']
    
    for c in continuous_ids:
        meta[c] = {'field_length': 1, 'type': 'continuous'}
    for c in categorical_ids:
        meta[c] = {'field_length': int(max(data_df[c+'_num'].values)) + 1, 'type': 'categorical'}
    
    meta['Bodytype'] = {'field_length': 13, 'type': 'categorical'}
    
    # Writing to sample.json
    with open(join(BASE, 'meta.json'), "w") as f:
        f.write(json.dumps(meta, indent=4) )

    