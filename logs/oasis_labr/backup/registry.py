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
    'clip_vitb16': clip_vit_b_16,
    'clip_vitb32': clip_vit_b_32,
    'clip_vitl32': clip_vit_l_32,
    'mm': MultimodalModel,
    'res_mm': ResMultimodalModel,
    'film': FilmHNN,
    'daft': DAFT,
    'ppnet': PPNet,
    'tabattn': ResNetTabAttention,
    'clip_resnet50_dvm': clip_resnet50_dvm,
    'mmdynamic': MMDynamicModel,
    'trimodal': TriModalModel,
    'tabtransformer': tabtransformer,
    'irene': IRene
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
        'acc@5': top5_accuracy,
        'f1': f1
}

DATASET = {
    'dvm': DVM,
    'preload_dvm': DVMPre,
    'preloadres_dvm': DVMPreRes,
    'dvm_lowregime': DVMLow,
    'preload_dvm_lowregime': DVMLowPre,
    'preloadres_dvm_lowregime': DVMLowPreRes,
    'covidar': COVID19AR,
    'preload_covidar': COVID19ARPre,
    'preloadres_covidar': COVID19ARPreRes,
    'mug': MUG, 
    'preload_mug': MUGPre,
    'isup': ISUP,
    'preload_isup': ISUPPre,
    'preloadres_isup': ISUPPreRes,
    'preload_mug_tri': MUGTriPre,
    'skin': Skin,
    'preload_skin': SkinPre,
    'res_skin': SkinRes,
    'oasis': Oasis,
    'preload_oasis': OasisPre,
    'res_oasis': OasisRes
}

Embedder = {
    'onehot': OneHotEmbedder,
    'text': TextEmbedder,
    'default': DefaultEmbedder,
    'split': SplitEmbedder
}

TABULAR_TRANSFORM = {
    'scarf': Scarf,
    'random_mask': RandomMask,
    'covidar_mapping': COVIDARMapping
}