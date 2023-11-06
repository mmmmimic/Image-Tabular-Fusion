import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import PurePath

df = pd.read_csv('/home/lmx/Image-Tabular-Fusion/data/prostate_ISUP/data_full.csv')
df = df[df['split']=='test']
df = df.reset_index()
print(len(df))
prob = np.load('tmp1.npy')
# subject = list(set(df['SubjectID'].values))
image_ids = list(map(lambda x: int(PurePath(x).parts[-1].split('_')[1]), df['Image_DIR'].values))
subject = df['SubjectID'].values
lesion = []
for i in range(len(subject)):
    lesion.append(subject[i] + f'_{image_ids[i]}')
df['LesionID'] = lesion
lesion = list(set(lesion))

preds = []
gts = []
for s in lesion:
    p = prob[df[df['LesionID']==s].index.values, :]
    p = p.mean(axis=0)
    p = np.argmax(p)
    preds.append(p)
    gts.append(df[df['LesionID']==s]['label'].values[0])

print(accuracy_score(gts, preds))

# image patient acc: 53.66
# gpt patient acc: 60.98