# train and test a model on a dataset
import torch
from data import get_dataset
import argparse
import torch.nn as nn
import torchvision.transforms as T
import yaml
import os
import model
from core import Trainer
from loss import get_criterion
from wrappers import wrapup_model

def read_configs(file_dir):
    if not os.path.exists(file_dir):
        raise 

def build_trainer(args, configs):
    pass

def backup():
    pass

def main(args, configs) -> int:
    print(args)
    print(configs)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multimodal classification'
    )

    parser.add_argument('--exp_name', type=str, metavar='-n', help="experiment name")
    parser.add_argument('--config', type=str, metavar='-c', help="configuration file path")
    parser.add_argument('--resume', type=str, metavar='-r', help="resume the training from <checkpoint> path")
    parser.add_argument('--train', type=bool, default=False, help="whether train the model")

    args = parser.parse_args()
    configs = read_configs(args.config)
    
    main(args, configs)
    sys.exit()
    
    from data import DVM, OneHotEmbedder, Scarf
    import torchvision.transforms as T
    

    model = nn.Sequential(
                    nn.Linear(63, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 286) # 286 categories in total
                        )
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    
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