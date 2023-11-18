from sklearn import metrics
import numpy as np
from functools import partial
from scipy.special import softmax

def accuracy(preds, scores, gts):
    # np.save('tmp1.npy', scores)
    return metrics.accuracy_score(gts.flatten(), preds.flatten())

def avg_accuracy(preds, scores, gts):
    return metrics.balanced_accuracy_score(gts.flatten(), preds.flatten())

def auc(preds, scores, gts):
    if scores.shape[1] > 1:
        softmax(scores, axis=-1)
    scores = scores[:,1]
    auc = metrics.roc_auc_score(gts.flatten(), scores, multi_class='ovo', average='macro', labels=np.arange(0, scores.shape[-1], 1, dtype=np.int64))
    if auc < 0.5:
        auc = 1- auc
    return auc

def topk_accuracy(preds, scores, gts, k=1):
    return metrics.top_k_accuracy_score(gts.flatten(), scores, k=k, labels=np.arange(0, scores.shape[-1], 1, dtype=np.int64))

top1_accuracy = partial(topk_accuracy, k=1)
top3_accuracy = partial(topk_accuracy, k=3)
top5_accuracy = partial(topk_accuracy, k=5)
top10_accuracy = partial(topk_accuracy, k=10)

    
