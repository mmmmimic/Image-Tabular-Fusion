import torch.nn as nn
from metrics import accuracy, avg_accuracy, auc, topk_accuracy
from wrappers import ModelWrapper, DictWrapper, TupleWrapper
from model import MLP, MLP2D 
from data import DVM
from data import OneHotEmbedder, TextEmbedder, DefaultEmbedder, Scarf, RandomMask


MODEL = {
    'mlp': MLP,
    'mlp2d': MLP2D
} 

LOSS = {
    'cse': nn.CrossEntropyLoss,
    'bce': nn.BCEWithLogitsLoss
}

METRICS = {
        'acc': accuracy,
        'avg_acc': avg_accuracy,
        'auc': auc,
        'acc@3': lambda x, y, z: topk_accuracy(x, y, z, k=3),
        'acc@5': lambda x, y, z: topk_accuracy(x, y, z, k=5)
}

WRAPPER = {
    'dict': DictWrapper,
    'tuple': TupleWrapper,
    'list': TupleWrapper,
    'default': ModelWrapper
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