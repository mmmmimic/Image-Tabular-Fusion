import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from sklearn.model_selection import train_test_split

### CONSTANTS ##############
ROOT = 'data/dvm_car'
CSV_DIR = join(ROOT, 'train_df_full.csv')
LABEL_DIR = join(ROOT, 'train_labels.npy')
############################

def low_data_split(df, nclasses):
    critical_ids = df.groupby('label', as_index=False).head(n=1)['index'].values
    ids = df['index'].values
    other_ids  = np.array(list(filter(lambda x: x not in critical_ids, ids)))
    
    to_fill_size = int(len(ids)*0.1) - len(critical_ids)
    stratify = df[df['index'].isin(other_ids)]['label'].values
    _, low_data_ids = train_test_split(other_ids, test_size=to_fill_size, random_state=2023, stratify=stratify)
    
    new_ids = np.concatenate([critical_ids,low_data_ids])
    
    return new_ids


if __name__ == "__main__":
    df = pd.read_csv(CSV_DIR)
    df = df.reset_index()
    label = np.load(LABEL_DIR)
    df['label'] = label
    nclasses = 286
    ids_01 = low_data_split(df, nclasses)
    df = df[df['index'].isin(ids_01)]
    ids_001 = low_data_split(df, nclasses)
    
    np.save(join(ROOT, 'ids_01.npy'), ids_01)
    np.save(join(ROOT, 'ids_001.npy'), ids_001)
    
