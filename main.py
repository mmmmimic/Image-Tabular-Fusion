# train and test a model on a dataset
import torch
from torch.utils.data import DataLoader
from data import DVM
import argparse
import torch.nn as nn
import torchvision.transforms as T
import json
import os
import model

parser = argparse.ArgumentParser(
    description='Multimodal segmentation'
)

parser.add_argument('--exp_name', type=str, metavar='-n', help="experiment name")
parser.add_argument('--config', type=str, metavar='-c', help="configuration file path")
parser.add_argument('--batch_size', type=int, default=16, metavar='-b')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epoch', type=float, default=1000)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--warmup', type=int, default=100)
parser.add_argument('--resume', type=str, metavar='-r', help="resume the training from <checkpoint> path")
parser.add_argument('--model', type=str, metavar='-m', help="model type from ['mlp']")
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--optimizer', type=str, default='sgd', help="optimizer from ['sgd', ]")
parser.add_argument('--augmentation', type=str)
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--tensorboard_log', type=bool, default=True, help="whether activate tensorboard logging")


args = parser.parse_args()

def build_model(args):
    model_type = 0


        # backup configs

def build_data(args):
    pass
    
model = 
optimizer = 
scheduler = 

print(args)


def train_one_epoch():
    "train the mdoel for one epoch"


if __name__ == "__main__":
    pass