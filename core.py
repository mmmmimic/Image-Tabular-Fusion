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

class ModelWrapper(nn.Module):
    """
    The wrapper wraps an arbitary deep neural network to the format fitting the trainer
    """
    def __init__(self, model, criterion=nn.CrossEntropyLoss) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
    
    def forward(self, data) -> Any:
        label = data['label']
        logit = self._forward(data)
        loss = criterion(logit, label)
        return loss
        
    def _forward(self, data):
        return self.model(data)
    
    def _backward(self):
        pass
    
    def __repr__(self) -> str:
        print(self.model)
        
    def _register_buffer(self, x):
        self.model.register_buffer(x)
    
class ImageWrapper(ModelWrapper):
    def __init__(self, model) -> None:
        super().__init__(model)

    def forward(self, data) -> Any:
        self._forward(data)
    
    def _compute_loss(self):
        pass    

class TabularWrapper(ModelWrapper):
    def __init__(self, model) -> None:
        super().__init__(model)

    def forward(self, data) -> Any:
        self._forward(data)
    
    def _compute_loss(self):
        pass    

class MultiModalWrapper(ModelWrapper):
    def __init__(self, model) -> None:
        super().__init__(model)

    def forward(self, data) -> Any:
        self._forward(data)
    
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
    def __init__(self, exp_name, config, model, tensorboard_log=True, checkpoint='', log_root='./logs') -> None:

        # initiate loggers        
        self.time_stamp = str(time.time()).replace('.', '_')
        self._initiate_loggers(log_root, exp_name, tensorboard_log)
        
        # read training hyperparams
        self.config = config
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.warmup_steps = config['warmup_steps']
        self.optimizer_type = config['optimizer_type']
        self.resampling = config['resampling']

        self.model = ModelWrapper(model)
        self.criterion = criterion
        
        self.global_step = 0
        self.epoch = 0 # epoch counter
        self.current_epoch = 0 # training start from this epoch
        
        self._build_optimizer()
        self._build_scheduler()
        
        # load checkpoint
        self.checkpoint = checkpoint
        self._resume_from(checkpoint)
        
    def _initiate_loggers(self, log_root, exp_name, tensorboard_log):
        # create folders to store trianing logs
        exp_dir = pth.join(log_root, exp_name)
        log_dir = pth.join(exp_dir, 'logs')
        training_log_dir = pth.join(exp_dir, f"log_{self.time_stamp}.log")
        
        print(f"Initiating logs at {exp_dir}...")
        print(f"Training logs will be saved at {training_log_dir}.")
        
        if not pth.exists(log_root):
           os.mkdir(log_root)
        if not pth.exists(exp_dir):
            os.mkdir(exp_dir)
        if not pth.exists(log_dir):
            os.mkdir(log_dir) 

        self.logger = Logger(file_path=training_log_dir, clear=False)
        
        self.tensorboard_log = tensorboard_log
        if self.tensorboard_log:
            self.board = SummaryWriter(log_dir=log_dir)
        
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.warmup_steps, T_mult=2, eta_min=1e-8, last_epoch=-1, verbose=True)

    def _validate(self, path):
        if not os.path.isfile(path):
            self.logger.fprint(f'{path} is not a file...')
            return False
        else:
            self.logger.fprint(f'Resuming training from {path}.')
            return True

    def _load(self):
        # load checkpoint
        raise NotImplementedError
    
    def _wrap_checkpoint(self):
        # update statedict
        checkpoint = {}

        # save training configs
        checkpoint['config'] = config

        #         
        
    def _save(self):
        # save checkpoint
        pass
        
    def _resume_from(self, checkpoint):
        # check if checkpoint is valid
        self._validate(checkpoint)
        
        raise NotImplementedError

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
        raise NotImplementedError
    
    def _validate_one_epoch(self):
        raise NotImplementedError
    
    def _train_one_iter(self):
        raise NotImplementedError
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
    
    def _validate_one_iter(self):
        with torch.no_grad():
            raise NotImplementedError
            self.global_step += 1
    
    def _log(self, msg, v):
        # log message in log and tensorboard
        self.logger.fprint(f"{msg} is {v}")
        self.board.add_scalar(tag=msg, scalar_value=v, global_step=self.global_step)
    
    def _draw_curves(self):
        raise NotImplementedError
    
    def _show_examples(self):
        raise NotImplementedError
    
    def _compute_loss(self):
        raise NotImplementedError
    
    def _save_checkpoint(self):
        raise NotImplementedError
    
    def fit(self, train_data, val_data=None):
        self._build_data_loader(train_data, val_data)
        
        for self.current_epoch in range(self.epochs):
            self._train_one_epoch()
            self._validate_one_epoch()
    
    def predict(self, test_data):
        self._build_data_loader(None, test_data)
    
    def resume(self):
        # resume from checkpoint
        raise NotImplementedError

if __name__ == "__main__":
    from data import DVM, RandomMask, OneHotEmbedder
    import torchvision.transforms as T
    
    exp_name = "test_exp"
    config = {
        'batch_size': 32,
        'epochs': 1000,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'warmup_steps': 1000,
        'checkpoint': None,
        'optimizer_type': 'adamw',
        'resampling': False
    }
    
    model = torch.nn.Linear(2048, 10)
    criterion = nn.CrossEntropyLoss()
    tab_transform = RandomMask(corrupt_rate=0.7)
    transforms = {
        'tab_tf': tab_transform, 
        'img_tf': T.Compose(
            [
                T.ToTensor()
            ]
        )
    }
    trainset = DVM(split='train', transforms=transforms, numerical=False, tab_embedder=OneHotEmbedder)
    valset = DVM(split='val', transforms=transforms, numerical=False, tab_embedder=OneHotEmbedder)
    
    trainer = Trainer(exp_name, config, model, tensorboard_log=True)
    trainer.fit(trainset, valset)
