import torch.nn as nn

def get_criterion(name, *args, **kwargs):
    if name == 'cse':
        return nn.CrossEntropyLoss(*args, **kwargs)
    elif name == 'bce':
        return nn.BCEWithLogitsLoss(*args, **kwargs)
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    criterion = get_criterion('cse', label_smoothing=0.1)
    print(criterion)