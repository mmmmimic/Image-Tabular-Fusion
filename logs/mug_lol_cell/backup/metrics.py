from sklearn import metrics
import numpy as np
from functools import partial
from scipy.special import softmax

def accuracy(preds, scores, gts):
    return metrics.accuracy_score(gts.flatten(), preds.flatten())

def avg_accuracy(preds, scores, gts):
    return metrics.balanced_accuracy_score(gts.flatten(), preds.flatten())

def auc(preds, scores, gts):
    if scores.shape[1] > 1:
        softmax(scores, axis=-1)
    scores = scores[:,1]
    return metrics.roc_auc_score(gts.flatten(), scores, multi_class='ovo', average='weighted', labels=np.arange(0, scores.shape[-1], 1, dtype=np.int64))

def topk_accuracy(preds, scores, gts, k=1):
    return metrics.top_k_accuracy_score(gts.flatten(), scores, k=k, labels=np.arange(0, scores.shape[-1], 1, dtype=np.int64))

top1_accuracy = partial(topk_accuracy, k=1)
top3_accuracy = partial(topk_accuracy, k=3)
top5_accuracy = partial(topk_accuracy, k=5)
top10_accuracy = partial(topk_accuracy, k=10)

    
