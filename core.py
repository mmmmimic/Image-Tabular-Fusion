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
    args:
        model (torch.nn.Module): 
        metric_names (list[str]): evaluation metrics, can be chosen from ['acc', 'avg_acc', 'auc', 'acc@3', 'acc@5']
        mode (str): 'accum' or 'avg'. If 'accum', the metric will be computed over the full dataset. 
        criterion (dict{'loss name': [weight, function]}): cost function dict
    """
    def __init__(self, model, criterion={'cse': [1, nn.CrossEntropyLoss()]}) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self._loss = 0
        
        self.loss_dict = {}
        for name in criterion.keys():
            self.loss_dict[name] = 0.
    
    def forward(self, data) -> Any:
        logit = self._forward(data)
        data['logit'] = logit
        
        data['loss_dict'] = self._compute_loss(data)
        data['loss'] = self.loss
        
        return data
        
    def _forward(self, data):
        return self.model(data)
    
    def _backward(self):
        self.model.backward()
    
    def _compute_loss(self, data):
        label = data['label']
        logit = data['logit']
        self._loss = 0
        for name in self.criterion.keys():
            _weight = self.criterion[name][0]
            _criterion = self.criterion[name][1]
            if isinstance(_criterion, nn.Module):
                # torch built-in cost function
                loss_val =  _criterion(logit, label)
            else:
                # custume cost function
                loss_val =  _criterion(data)
            self.loss_dict[name] = loss_val
            self._loss += _weight * loss_val

    def __repr__(self) -> str:
        print(self.model)
        
    def _register_buffer(self, x):
        self.model.register_buffer(x)
    
    @property
    def loss(self):
        # cost function
        return self._loss
    

# class ImageWrapper(ModelWrapper):
#     def __init__(self, model) -> None:
#         super().__init__(model)

#     def _forward(self, data) -> Any:
#         return self.model(data['image'])

# class TabularWrapper(ModelWrapper):
#     def __init__(self, model) -> None:
#         super().__init__(model)

#     def _forward(self, data) -> Any:
#         return self.model(data['tab_line'])

# class MultiModalWrapper(ModelWrapper):
#     def __init__(self, model) -> None:
#         super().__init__(model)

#     def _forward(self, data) -> Any:
#         return self.model(data['image'], data['tab_line'])
     
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
    def __init__(self, exp_name, config, model, tensorboard_log=True, checkpoint_path=None, log_root='./logs') -> None:

        # initiate loggers        
        self.time_stamp = str(time.time()).replace('.', '_')
        self.exp_dir = pth.join(log_root, exp_name)
        self._initiate_loggers(log_root, tensorboard_log)
        
        # read training hyperparams
        self.config = config
        self._initiate_hyperparams()
        
        self.model = ModelWrapper(model)
        
        self.global_step = 0
        self.current_epoch = 0 # epoch counter
        self.start_epoch = 0 # training start from this epoch
        
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
        self.accum_step = config['accum_step']
        self.device = torch.device(config['device'])
        self.mode = config['mode']
        self.metric_names = config['metric_names']
        self.metric_monitor = MetricManager(self.metric_names, self.mode)
        

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
                for key in ['config', 'state_dict', 'criterion', 'global_step', 'current_epoch', 'optimizer', 'scheduler']:
                    assert key in self.checkpoint.keys()
                return True
            except:
                self.logger.fprint(f'{path} is broken.')
                return False

    def _load(self):
        # load checkpoint
        raise NotImplementedError
    
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
        checkpoint['optimizer_state'] = self.optimizer.state_dict()
        checkpoint['scheduler_state'] = self.scheduler.state_dict()
        
        self.checkpoint = checkpoint
    
    def _wrap_and_save(self, mode):
        assert mode in ['checkpoint', 'best_model', 'last_model']
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

            self.model = self.model.load_state_dict(self.checkpoint['state_dict'])
            self.criterion = self.checkpoint['criterion']
            
            self.global_step = self.checkpoint['global_step']
            self.current_epoch = self.checkpoint['current_epoch'] # epoch counter
            self.start_epoch = self.checkpoint['current_epoch'] # training start from this epoch
            
            self._build_optimizer()
            self._build_scheduler()
            
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(self.checkpoint['scheduler_state'])
            
        else:
            self.checkpoint = {}
            self.logger.fprint('Training from scratch.')

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
        data['metric_dict'] = self.metric_monitor.metric_values
        data['metric_report'] = self.metric_monitor.summarization
        raise NotImplementedError
    
    def _validate_one_epoch(self):
        raise NotImplementedError
    
    def _train_one_iter(self):
        raise NotImplementedError
        self.metric_monitor.update(logit, label)
        if not self.global_step % self.accum_step:
            self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        
    
    def _validate_one_iter(self):
        with torch.no_grad():
            raise NotImplementedError
            self.global_step += 1
    
    def _save_var(self, msg, v):
        # save a variable, log message in log and tensorboard
        self.logger.fprint(f"{msg} is {v}")
        self.board.add_scalar(tag=msg, scalar_value=v, global_step=self.global_step)
    
    def _draw_curves(self):
        raise NotImplementedError
    
    def _show_examples(self):
        raise NotImplementedError
    
    def _compute_loss(self):
        return self.model.loss
    
    def fit(self, train_data, val_data=None):
        try:
            self._build_data_loader(train_data, val_data)
            
            for self.current_epoch in range(self.start_epoch, self.epochs):
                self._train_one_epoch()
                self._validate_one_epoch()
        except KeyboardInterrupt:
            self._wrap_and_save('checkpoint')
    
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
        'resampling': False,
        'device': 'cuda'
    }
    
    model = torch.nn.Linear(2048, 10)
    criterion = {'cse': nn.CrossEntropyLoss()}
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
    
    trainer = Trainer(exp_name, config, model, criterion, tensorboard_log=True)
    trainer.fit(trainset, valset)
    
    print('-------------------------------------------')
