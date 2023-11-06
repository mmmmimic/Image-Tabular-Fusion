import torch.nn as nn
from metrics import *
from model import *
from data import *

MODEL = {
    'mlp': MLP,
    'mlp2d': MLP2D,
    'clipmlp': ClipMLP,
    'resnet50': resnet50,
    'clip_resnet50': clip_resnet50,
    'vitb16': vit_b_16,
    'mm': MultimodalModel,
    'res_mm': ResMultimodalModel,
    'film': FilmHNN,
    'daft': DAFT,
    'ppnet': PPNet,
    'tabattn': ResNetTabAttention,
    'clip_resnet50_dvm': clip_resnet50_dvm,
    'mmdynamic': MMDynamicModel,
    'trimodal': TriModalModel
} 

class MM_Loss():
    def __call__(self, x):
        return x['loss']
    
LOSS = {
    'cse': nn.CrossEntropyLoss,
    'bce': nn.BCEWithLogitsLoss,
    'mmdynamic': MM_Loss
}

METRICS = {
        'acc': accuracy,
        'avg_acc': avg_accuracy,
        'auc': auc,
        'acc@3': top3_accuracy,
        'acc@5': top5_accuracy
}

DATASET = {
    'dvm': DVM,
    'preload_dvm': DVMPre,
    'preloadres_dvm': DVMPreRes,
    'dvm_lowregime': DVMLow,
    'preload_dvm_lowregime': DVMLowPre,
    'covidar': COVID19AR,
    'preload_covidar': COVID19ARPre,
    'mug': MUG, 
    'preload_mug': MUGPre,
    'isup': ISUP,
    'preload_isup': ISUPPre,
    'preload_mug_tri': MUGTriPre
}

Embedder = {
    'onehot': OneHotEmbedder,
    'text': TextEmbedder,
    'default': DefaultEmbedder
}

TABULAR_TRANSFORM = {
    'scarf': Scarf,
    'random_mask': RandomMask,
    'covidar_mapping': COVIDARMapping
}