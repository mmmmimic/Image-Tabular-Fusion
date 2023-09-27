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
from tqdm import tqdm
import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

class ModelWrapper(nn.Module):
    """
    The wrapper wraps an arbitary deep neural network to the format fitting the trainer
    args:
        model (torch.nn.Module): 
        criterion (dict{'loss name': [weight, function]}): cost function dict
    """
    def __init__(self, model, device) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.model.to(device)
    
    def forward(self, data) -> Any:
        outputs = self._forward(data)
        if isinstance(outputs, dict):
            return outputs
        else:
            logit = outputs
            data['logit'] = logit
            return data
        
    def _forward(self, data):
        return self.model(data.to(self.device))

    def __repr__(self) -> str:
        return(str(self.model))
        
    def _register_buffer(self, x):
        self.model.register_buffer(x)

class ImageWrapper(ModelWrapper):
    def __init__(self, model, device) -> None:
        super().__init__(model, device)

    def _forward(self, data) -> Any:
        return self.model(data['image'].to(self.device))

class TabularWrapper(ModelWrapper):
    def __init__(self, model, device) -> None:
        super().__init__(model, device)

    def _forward(self, data) -> Any:
        return self.model(data['tab_line'].to(self.device))

class MultiModalWrapper(ModelWrapper):
    def __init__(self, model, device) -> None:
        super().__init__(model, device)

    def _forward(self, data) -> Any:
        return self.model(data['image'].to(self.device), data['tab_line'].to(self.device))
    
class StandardWrapper(ModelWrapper):
    def __init__(self, model, device) -> None:
        super().__init__(model, device)
    
    def forward(self, data) -> Any:
        image, label = data
        logit = self._forward(image)
        return {'image': image, 'label': label, 'logit': logit}
            
class Trainer:
    """
    The trainer trains or validates an arbitrary model. 
    args:
        exp_name (str): experiment name, which decides the log path
        config (dict): configurations of model training
        model (torch.nn.Module): deep neural network that is going to be trained
        tensorboard_log (bool): if activate tensorboard
        checkpoint_path (str | None): checkpoint path, None if training from scratch
        log_root (str): root path of the training logs 
    """
    def __init__(self, exp_name, config, model, tensorboard_log=True, checkpoint_path=None, log_root='./logs', wrapper='image', *args, **kwargs) -> None:

        # initiate loggers        
        self.time_stamp = str(time.time()).replace('.', '_')
        self.exp_dir = pth.join(log_root, exp_name)
        self._initiate_loggers(log_root, tensorboard_log)
        
        # read training hyperparams
        self.config = config
        self._initiate_hyperparams()
        
        # wrap up the model
        self.model = model
        self._initiate_wrapper(wrapper)
        
        self.global_step = 0
        self.current_epoch = 0 # epoch counter
        self.start_epoch = 0 # training start from this epoch
        self.best_epoch = 0
        self.best_metric = 0.
        
        self._build_optimizer()
        self._build_scheduler()
        
        # load checkpoint
        self.checkpoint_path = checkpoint_path
        self._resume_from(checkpoint_path)   
        
    def _initiate_hyperparams(self):
        config = self.config
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.warmup_steps = config['warmup_steps']
        self.optimizer_type = config['optimizer_type']
        self.resampling = config['resampling']
        self.device = torch.device(config['device'])
        
        self.accum_step = config['accum_step']
        self.mode = config['mode']
        self.metric_names = config['metric_names']        
        self.criterion = config['criterion']
        
        self.monitor_metric = config['monitor_metric']
        
    def _initiate_loggers(self, log_root, tensorboard_log):
        # create folders to store trianing logs
        log_dir = pth.join(self.exp_dir, 'logs')
        training_log_dir = pth.join(self.exp_dir, f"log_{self.time_stamp}.log")
        
        print(f"Initiating logs at {self.exp_dir}...")
        print(f"Training logs will be saved at {training_log_dir}.")
        
        if not pth.exists(log_root):
           os.mkdir(log_root)
        if not pth.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        if not pth.exists(log_dir):
            os.mkdir(log_dir) 

        self.logger = Logger(file_path=training_log_dir, clear=False)
        
        self.tensorboard_log = tensorboard_log
        if self.tensorboard_log:
            self.board = SummaryWriter(log_dir=log_dir)

    def _initiate_wrapper(self, wrapper):
        if wrapper == 'image':
            self.model = ImageWrapper(model, self.device)       
        elif wrapper == 'tabular':
            self.model = TabularWrapper(model, self.device)
        elif wrapper == 'multimodal':
            self.model = MultiModalWrapper(model, self.device)
        elif wrapper == 'standard':
            self.model = StandardWrapper(model, self.device)
        else:
            self.model = ModelWrapper(model, self.device)   
        
        self.logger.fprint('Model structure: ')    
        self.logger.fprint(self.model.__repr__())  
        
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.warmup_steps, T_mult=2, eta_min=1e-8, last_epoch=-1, verbose=False)

    def _validate_and_load(self, path):
        if path is None:
            return False
        elif not os.path.isfile(path):
            self.logger.fprint(f'{path} is not a file...')
            return False
        else:
            self.logger.fprint(f'Resuming training from {path}.')
            try:
                self.checkpoint = torch.load(path, map_location=self.device)
                for key in ['config', 'state_dict', 'criterion', 'global_step', 'current_epoch', 'optimizer_state', 'scheduler_state', 'best_epoch', 'best_metric']:
                    assert key in self.checkpoint.keys()
                return True
            except:
                self.logger.fprint(f'{path} is broken.')
                return False
    
    def _wrap_checkpoint(self):
        # update statedict
        checkpoint = {}

        # save training configs
        checkpoint['config'] = self.config # hyperparams are stored here
        
        # save model parameters
        checkpoint['state_dict'] = self.model.state_dict()
        checkpoint['criterion'] = self.criterion
        
        # save optimizers and schedulers
        checkpoint['global_step'] = self.global_step
        checkpoint['current_epoch'] = self.current_epoch
        checkpoint['best_epoch'] = self.best_epoch
        checkpoint['best_metric'] = self.best_metric
        checkpoint['optimizer_state'] = self.optimizer.state_dict()
        checkpoint['scheduler_state'] = self.scheduler.state_dict()
        
        self.checkpoint = checkpoint
    
    def _wrap_and_save(self, mode):
        assert mode in ['checkpoint', 'best_model', 'last_model', 'backup']
        self._wrap_checkpoint()
        save_folder = pth.join(self.exp_dir, 'models')
        if not pth.isdir(save_folder):
            os.mkdir(save_folder)
        name = f"{mode}_{str(time.time()).replace('.', '_')}_epoch{self.current_epoch}.t7"
        save_path = pth.join(save_folder, name)
        torch.save(self.checkpoint, save_path)
        
    def _resume_from(self, checkpoint_path):
        # check if checkpoint is valid
        if self._validate_and_load(checkpoint_path):
            # read training hyperparams
            self.config = self.checkpoint['config']
            self._initiate_hyperparams()

            self.model.load_state_dict(self.checkpoint['state_dict'])
            self.criterion = self.checkpoint['criterion']
            
            self.global_step = self.checkpoint['global_step']
            self.current_epoch = self.checkpoint['current_epoch'] # epoch counter
            self.start_epoch = self.checkpoint['current_epoch'] # training start from this epoch
            self.best_epoch = self.checkpoint['best_epoch']
            self.best_metric = self.checkpoint['best_metric']
            
            self._build_optimizer()
            self._build_scheduler()
            
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(self.checkpoint['scheduler_state'])
            
        else:
            self.checkpoint = {}
            
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
    
    def _prepare_monitors(self):
        self.loss = AverageMeter()
        self.loss_dict = {}
        self.loss_monitor = {}
        for name in criterion.keys():
            self.loss_monitor[name] = AverageMeter()
            self.loss_dict[name] = 0.
            
        self.metric_monitor = MetricManager(self.metric_names, self.mode)

    def _train_one_epoch(self):
        # train
        self.model.train()
        self._prepare_monitors()
        
        for self.data in tqdm(self.train_loader):
            self._train_one_iter()
            
        train_metric_dict = self.metric_monitor.metric_values
        train_metric_report = self.metric_monitor.summarization
        train_loss_dict = self.loss_dict
        train_loss_report = self._report_loss()  
                    
        # saving metrics and losses to tensorboard
        self._save_dict(train_metric_dict, 'train')
        self._save_dict(train_loss_dict, 'train')        
            
        msg = f'epoch {self.current_epoch}' + '\n' + '<--[TRAINING]--> ' + train_loss_report + '\n' + train_metric_report
        self.logger.fprint(msg)
        
    def _validate_one_epoch(self):
        # validation    
        self.model.eval()
        self._prepare_monitors()
        for self.data in tqdm(self.test_loader):
            self._validate_one_iter()
               
    def _train_one_iter(self):
        self.optimizer.zero_grad()
        self.data = self.model(self.data)
        
        logit, label = self.data['logit'], self.data['label']
        
        self.metric_monitor.update(logit, label)
        
        self._compute_loss()
                
        if not self.global_step % self.accum_step:
            self._save_var('loss_train', self.loss.avg)
            self.loss.sum.backward()
            self.optimizer.step()
            self.loss._reset()
            
        self.scheduler.step() # for each step, update the scheduler
        
        self.global_step += 1
         
    def _validate_one_iter(self):
        with torch.no_grad():
            self.data = self.model(self.data)
            
            logit, label = self.data['logit'], self.data['label']
            
            self.metric_monitor.update(logit, label)
            self._compute_loss()
                        
    def _save_dict(self, d, prefix):
        for name in d.keys():
            self._save_var(name + f'_{prefix}', d[name])
    
    def _save_var(self, name, value):
        # save a variable, log message in log and tensorboard
        if self.tensorboard_log:
            self.board.add_scalar(tag=name, scalar_value=value, global_step=self.global_step)
    
    def _show_examples(self):
        raise NotImplementedError
    
    def _compute_loss(self):
        label = self.data['label'].to(self.device)
        logit = self.data['logit']
        batch_size = logit.size(0)
        _loss = 0.
        for name in self.criterion.keys():
            _weight = self.criterion[name][0]
            _criterion = self.criterion[name][1]
            if isinstance(_criterion, nn.Module):
                # torch built-in cost function
                loss_val =  _criterion(logit, label)
            else:
                # custume cost function
                loss_val =  _criterion(self.data)
            self.loss_monitor[name].update(loss_val.item(), batch_size)
            _loss += _weight * loss_val
            self.loss_dict[name] = self.loss_monitor[name].avg
        
        self.loss.update(_loss, batch_size)
            
    def _report_loss(self):
        loss_report = ''
        for name in self.loss_dict.keys():
            loss_report += f'{name}: {self.loss_dict[name]:.4f}, '
        loss_report = loss_report[:-2]
        return loss_report
    
    def fit(self, train_data, val_data=None):
            self._build_data_loader(train_data, val_data)
            
            for self.current_epoch in range(self.start_epoch, self.epochs):
                self._train_one_epoch()
                self._validate_one_epoch()
                
                eval_metric_dict = self.metric_monitor.metric_values
                eval_metric_report = self.metric_monitor.summarization
                eval_loss_dict = self.loss_dict 
                eval_loss_report = self._report_loss()
                
                # saving metrics and losses to tensorboard
                self._save_dict(eval_metric_dict, 'eval')
                self._save_dict(eval_loss_dict, 'eval')
                self._save_var('loss_eval', self.loss.avg)
                
                msg  = '<--[VALIDATION]--> ' + eval_loss_report + '\n' + eval_metric_report
                self.logger.fprint(msg)
                
                # save the best model
                if eval_metric_dict[self.monitor_metric] > self.best_metric:
                    self.best_metric = eval_metric_dict[self.monitor_metric]
                    self.best_epoch = self.current_epoch
                    self.logger.fprint(f'[MODEL SAVED] epoch {self.best_epoch}, with {self.monitor_metric} = {self.best_metric:.4f}.')
                    # remove the previous best model
                    os.system(f"rm {pth.join(self.exp_dir, 'models', 'best_model*')}")
                    self._wrap_and_save(mode='best_model')
                
                self.logger.fprint(f'Best {self.monitor_metric}: {self.best_metric:.4f} at epoch {self.best_epoch}')  
                              
                if not self.current_epoch % 100:
                    self.logger.fprint(f'[CHECKPOINT SAVED] epoch {self.current_epoch}.')
                    self._wrap_and_save(mode='checkpoint')        
                    
                self.logger.fprint('\n')

            self.logger.fprint(f'[MODEL SAVED] epoch {self.current_epoch}.')
            self._wrap_and_save(mode='last_model')
    
    def predict(self, test_data):
        self.logger.fprint('Start validation.')
        self._build_data_loader(None, test_data)
        self._validate_one_epoch()
        
        # report result
        eval_metric_report = self.metric_monitor.summarization
        eval_loss_report = self._report_loss()
        
        msg  = '<--[TEST]--> ' + eval_loss_report + '\n' + eval_metric_report
        self.logger.fprint(msg)
        

if __name__ == "__main__":
    from data import DVM, OneHotEmbedder, Scarf
    import torchvision.transforms as T
    
    exp_name = "test_exp"
    config = {
        'batch_size': 128,
        'epochs': 1000,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'warmup_steps': 1000,
        'checkpoint': None,
        'optimizer_type': 'adamw',
        'resampling': False,
        'device': 'cuda',
        'accum_step': 1, # update model parameters in each iteration
        'mode': 'accum',
        'metric_names': ['acc', 'avg_acc'],
        'criterion': {'cse': [1, nn.CrossEntropyLoss()]},
        'monitor_metric': 'acc'
    }
    
    model = nn.Sequential(
                    nn.Linear(63, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 286) # 286 categories in total
                        )
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    
    criterion = {'cse': nn.CrossEntropyLoss()}
    
    tab_transform = Scarf(corrupt_rate=0.7)
    transforms = {
        # 'tab_tf': tab_transform,
        'tab_tf': lambda x, y: y,  
        'img_tf': T.Compose(
            [
                T.ToTensor()
            ]
        )
    }
    trainset = DVM(split='train', transforms=transforms, numerical=True, tab_embedder=OneHotEmbedder())
    valset = DVM(split='val', transforms=transforms, numerical=True, tab_embedder=OneHotEmbedder())
    testset = DVM(split='test', transforms=transforms, numerical=True, tab_embedder=OneHotEmbedder())
    
    # import torchvision
    # trainset = torchvision.datasets.CIFAR10(root='.', train=True, transform=T.ToTensor(), download=False)
    # valset = torchvision.datasets.CIFAR10(root='.', train=False, transform=T.ToTensor(), download=False)
    
    trainer = Trainer(exp_name, config, model, tensorboard_log=False, checkpoint_path='logs/test_exp/models/best_model_1695785721_9514947_epoch317.t7', log_root='./logs', wrapper='tabular')
    # trainer.fit(trainset, valset)
    trainer.predict(testset)
    
    print('-------------------------------------------')
