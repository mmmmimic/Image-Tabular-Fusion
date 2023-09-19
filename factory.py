from typing import Any
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import Logger
from metrics import MetricManager, AverageMeter
import os.path as pth
import os
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
import time
import torch.nn as nn


class ModelWrapper:
    """
    The wrapper wraps an arbitary deep neural network to the format fitting the trainer
    """
    def __init__(self, model, criterion=nn.CrossEntropyLoss) -> None:
        self.model = model
        self.criterion = criterion
    
    def __call__(self, data) -> Any:
        label = data['label']
        logit = self._forward(data)
        loss = criterion(logit, label)
        
    
    def _forward(self, data):
        return self.model(data)
    
    def _backward(self):
        pass
    
    def __repr__(self) -> str:
        print(self.model)

class ImageWrapper(ModelWrapper):
    def __init__(self, model) -> None:
        super().__init__(model)

    def __call__(self, data) -> Any:
        self._forward(data)
    
    def _forward(self):
        pass
    
    def _backward(self):
        pass
    
    def _compute_loss(self):
        pass    
            

class Trainer:
    """
    The trainer trains or validates an arbitrary model. 
    args:
        exp_name: experiment name, which decides the log path
        config: configurations of model training
        model: deep neural network that is going to be trained
    """
    def __init__(self, exp_name, config, model, tensorboard_log=True) -> None:
        
        # create folders to store trianing logs
        log_root = './logs'
        exp_dir = pth.join(log_root, exp_name)
        log_dir = pth.join(exp_dir, 'logs')
        training_log_dir = pth.join(exp_dir, f"log_{str(time.time()).replace('.', '_')}.log")
        
        if not pth.exists(log_root):
           os.mkdir(log_root)
        if not pth.exists(exp_dir):
            os.mkdir(exp_dir)
        if not pth.exists(log_dir):
            os.mkdir(log_dir) 
        
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.warmup_steps = config.warmup
        self.checkpoint = config.resume
        self.optimizer_type = config.optimizer_type
        self.logger = Logger(file_path=training_log_dir, clear=False)
        
        
        self.tensorboard_log = tensorboard_log
        if self.tensorboard_log:
            self.board = SummaryWriter(log_dir=log_dir)
            
        self.checkpoint = None
    
        self.model = ModelWrapper(model)
        self.criterion = criterion
        
        self.global_step = 0
        self.current_epoch = 0
        
        self._build_optimizer()
        self._build_scheduler()
        
    def _build_optimizer(self):
        if self.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NameError
        
    def _build_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.warmup_steps, T_mult=1.1, eta_min=1e-8, last_epoch=-1, verbose=True)

    def _build_data_loader(self, train_data, val_data, **kwargs):
        if train_data is not None:
            if self.resampling:
                self.train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, sampler=ImbalancedDatasetSampler(train_data), **kwargs)
            else:
                self.train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, **kwargs)
        else:
            self.train_loader = None
                
        if val_data is not None:
            self.test_loader = DataLoader(dataset=val_data, batch_size=self.batch_size, shuffle=False, **kwargs)
        else:
            self.test_loader = None
    
    def _train_one_epoch(self):
        pass
    
    def _validate_one_epoch(self):
        pass
    
    def _train_one_iter(self):
        pass
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
    
    def _validate_one_iter(self):
        with torch.no_grad():
            pass
            self.global_step += 1
    
    def _log(self, msg, v):
        # log message in log and tensorboard
        self.logger.fprint(f"{msg} is {v}")
        self.board.add_scalar(tag=msg, scalar_value=v, global_step=self.global_step)
    
    def _draw_curves(self):
        pass
    
    def _draw_example(self):
        pass
    
    def _compute_loss(self):
        pass
    
    def _save_checkpoint(self):
        pass
    
    def fit(self, train_data, val_data):
        self._build_data_loader(train_data, val_data)
        for self.current_epoch in self.epochs:
            self._train_one_epoch()
            self._validate_one_epoch()
    
    def predict(self, test_data):
        self._build_data_loader(None, test_data)
    
    def resume(self):
        # resume from checkpoint
        pass

if __name__ == "__main__":
    exp_name = "test"
    configs = {
        
    }
    model = torch.nn.Linear(2048, 10)
    criterion = nn.CrossEntropy()
    trainer = Trainer(exp_name, config, model, tensorboard_log=True)
