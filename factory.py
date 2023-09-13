import torch

class Trainer:
    """
    args:

    """
    def __init__(self, configs, model) -> None:
        self.batch_size = configs.batch_size
        self.lr = configs.lr
        self.weight_decay = configs.weight_decay
        self.warmup = configs.warmup
        self.resume = configs.resume
        
        self.model = model
        
    def _train_one_epoch(self):
        pass
    
    def _validate_one_epoch(self):
        pass
    
    def _train_one_iter(self):
        pass
    
    def _validate_one_iter(self):
        pass
    
    def fit(self):
        pass
    
    def predict(self):
        pass

if __name__ == "__main__":
    pass
