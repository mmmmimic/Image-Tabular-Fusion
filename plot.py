import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import json
from sklearn.manifold import TSNE
test_emb = np.load('/home/lmx/Image-Tabular-Fusion/data/dvm_car/clip_gpt_test.npy')
learn_emb = np.load('/home/lmx/Image-Tabular-Fusion/dvm_tab_emb.npy')
df = pd.read_csv('/home/lmx/Image-Tabular-Fusion/data/dvm_car/test_df_full.csv')
def plot(emb_item):
    key = emb_item[0]
    value = emb_item[1]
    scale = df[key].values
    # do pca
#     pca = PCA(n_components=3, svd_solver='full')
#     pca.fit(value)
#     value = pca.transform(value)
#     value = TSNE(n_components=2, learning_rate='auto',
#                 init='random', perplexity=3).fit_transform(value)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    index = np.abs(value[:,2])<1
    value = value[index,:]
    scale = scale[index]
    index = np.abs(value[:,1])<1
    value = value[index,:]
    scale = scale[index]
    index = np.abs(value[:,0])<1
    value = value[index,:]
    scale = scale[index]
    ax.scatter(value[:,0], value[:,1], value[:,2], c=scale)
    # plt.title(f"CLIP embedding of attribute '{key}'")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    # ax.axes.set_xlim3d(left=-1, right=1)
    # ax.axes.set_ylim3d(bottom=-1, top=1)
    # ax.axes.set_zlim3d(bottom=-1, top=1)
    plt.tight_layout()
#     plt.colorbar()
    plt.show()

#     plt.figure()
#     plt.scatter(value[:,0], value[:,1], c=scale)
#     plt.colorbar()
#     plt.title(f'CLIP embedding of attribute {key}')
#     plt.show()
plot(('Color_num', test_emb[:,0,...]))
plot(('Color_num', learn_emb[:,0,...]))