import torch.nn as nn
from metrics import *
from model import *
from data import *

MODEL = {
    'mlp': MLP,
    'mlp2d': MLP2D,
    'resnet50': resnet50,
    'vitb16': vit_b_16
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
    'dvm': DVM
}

Embedder = {
    'onehot': OneHotEmbedder,
    'text': TextEmbedder,
    'default': DefaultEmbedder
}

TABULAR_TRANSFORM = {
    'scarf': Scarf,
    'random_mask': RandomMask
}