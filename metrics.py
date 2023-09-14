from sklearn import metrics
import torch
import numpy as np

def accuracy(preds, scores, gts):
    return metrics.accuracy_score(gts.flatten(), preds.flatten())

def avg_accuracy(preds, scores, gts):
    return metrics.balanced_accuracy_score(gts.flatten(), preds.flatten())

def auc(preds, scores, gts):
    return metrics.roc_auc_score(gts.flatten(), scores, multi_class='ovo', average='weighted', labels=np.arange(0, scores.shape[-1], 1, dtype=np.int64))

def topk_accuracy(preds, scores, gts, k=1):
    return metrics.top_k_accuracy_score(gts.flatten(), scores, k=k, labels=np.arange(0, scores.shape[-1], 1, dtype=np.int64))

class AverageMeter:
    def __init__(self) -> None:
        self._reset()
        
    def _reset(self):
        self._val = 0
        self._sum = 0
        self._avg = 0
        self._count = 0
        
    def update(self, val, n=1):
        self._val = val
        self._sum += val*n
        self._count += n
        self._avg = self._sum / self._count
    
    @property
    def avg(self):
        return self._avg
    
class MetricManager:
    def __init__(self, metric_names, mode='avg') -> None:
        assert mode in ['avg', 'accum']
        self.metric_names = metric_names
        self.mode = mode
        
        self.preds = []
        self.gts = []
        self.logits = []
        
        self.val = {}
        for name in metric_names:
            if mode  == 'avg':
                self.val[name] = AverageMeter()
            elif mode == 'accum':
                self.val[name] = 0.
            else:
                raise NameError
            
        self.map_dict = {
            'acc': lambda x, y, z: accuracy(x, y, z),
            'avg_acc': lambda x, y, z: avg_accuracy(x, y, z),
            'auc': lambda x, y, z: auc(x, y, z),
            'acc@3': lambda x, y, z: topk_accuracy(x, y, z, k=3),
            'acc@5': lambda x, y, z: topk_accuracy(x, y, z, k=5)
        }
        
    def __repr__(self) -> str:
        print(self.metric_values)
    
    @staticmethod
    def detach(tensor):
        return tensor.detach().cpu().numpy()
    
    def update(self, logits, gts):
        logits = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        if self.mode == 'avg':
            self.preds = self.detach(preds)
            self.logits = self.detach(logits)
            self.gts = self.detach(gts)
        elif self.mode == 'accum':
            self.gts = [np.concatenate(self.gts.append(self.detach(gts)), axis=0)]
            self.logits = [np.concatenate(self.logits.append(self.detach(logits)), axis=0)]
            self.preds = [np.concatenate(self.preds.append(self.detach(preds)), axis=0)]
        
        batch_size = preds.shape[0]
        
        metric_vals = self.get_metric()
        
        for name, val_ in zip(self.metric_names, metric_vals):
            if self.mode == 'avg':
                self.val[name].update(val_, n=batch_size)
            elif self.mode == 'accum':
                self.val[name] = val_
        
    def get_metric(self):
        if self.mode == 'avg':
            preds, logits, gts = self.preds, self.logits, self.gts
        elif self.mode == 'accum':
            preds, logits, gts = self.preds[0], self.logits[0], self.gts[0]
        metric_vals = map(lambda name: self.map_dict[name](preds, logits, gts), self.metric_names)
            
        return metric_vals
    
    @property
    def metric_values(self):
        vals = {}
        if self.mode == 'avg':
            for name in self.metric_names:
                vals[name] = self.val[name].avg
        elif self.mode == 'accum':
            vals = self.val
        return vals
    
    @property
    def summarization(self):
        # return a string containing all the metric values
        s = ''
        for v in self.metric_values:
            s += f"{v}: {self.metric_values[v]}"
            s += ', '
        s = s[:-2]
        return s
    
if __name__ == "__main__":
    meter = MetricManager(metric_names=['acc', 'auc', 'avg_acc', 'acc@5'], mode='avg')
    logits = torch.rand(32, 100)
    gts = torch.rand(32, 100)
    gts = torch.argmax(gts, dim=1)
    print(torch.argmax(logits, dim=1), gts)
    meter.update(logits, gts)
    print(meter.metric_values)
    print(meter.summarization)