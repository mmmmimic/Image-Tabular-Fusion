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
    'ppnet': PPNet
} 

LOSS = {
    'cse': nn.CrossEntropyLoss,
    'bce': nn.BCEWithLogitsLoss
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
    'covidar': COVID19AR,
    'preload_covidar': COVID19ARPre,
    'mug': MUG, 
    'preload_mug': MUGPre,
    'isup': ISUP,
    'preload_isup': ISUPPre
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