# train and test a model on a dataset
import argparse
import yaml
import os
from pathlib import PurePath
from glob import glob
from core import Trainer
from builder import build_criterion, build_dataset, build_model, build_trainer, wrap_model
from utils import create_folder


def read_configs(file_dir):
    with open(file_dir, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_file_name(full_dir):
    return PurePath(full_dir).parts[-1]

def save_folder(folder, dst_path):
    scripts = glob(os.path.join(folder, '*.py'))
    for s in scripts:
        os.system(f"cp {s} {os.path.join(dst_path, get_file_name(s))}")

def backup(exp_dir, config_dir):
    backup_folder = os.path.join(exp_dir, 'backup')
    create_folder(backup_folder)
    model_folder = os.path.join(backup_folder, 'model')
    create_folder(model_folder)
    config_folder = os.path.join(backup_folder, 'configs')
    create_folder(config_folder)
    data_folder = os.path.join(backup_folder, 'data')
    create_folder(data_folder)
    
    for s in ['builder.py', 'meters.py', 'metrics.py', 'registry.py', 'utils.py', 'wrappers.py']:
        os.system(f"cp {s} {os.path.join(backup_folder, s)}")
    
    save_folder('./model', model_folder)
    save_folder('./data', data_folder)
    os.system(f"cp {config_dir} {os.path.join(config_folder, get_file_name(config_dir))}")
    
    print(f"Backup saved at {backup_folder}.")
    

def main(args, configs) -> int:
    exp_name = args.exp_name
    train_config = configs['train']
    data_config = configs['data']
    model_config = configs['model']
    
    model = build_model(model_config['name'], model_config=model_config['config'])
    model = wrap_model(model, wrapper_type=model_config['wrapper'], device=train_config['device'])
    
    trainset, valset, testset = build_dataset(name=data_config['dataset'], config=data_config['config'])
    
    crtn_dict = train_config['criterion']
    criterion = build_criterion(crtn_dict)
    
    # train
    checkpoint_path = args.resume
    trainer = Trainer(exp_name, train_config, model, tensorboard_log=args.tensorboard_off, checkpoint_path=checkpoint_path, log_root='./logs', criterion=criterion)
    
    exp_dir = trainer.exp_dir
    config_dir = args.config
    backup(exp_dir, config_dir)
    
    if args.train:
        print('Train the model...')
        if checkpoint_path == 'last':
            trainer.resume(trainset, valset)
        else:
            trainer.fit(trainset, valset)
    else:
        print('Evaluate the model')
        if checkpoint_path == 'best' or checkpoint_path == '':
            # by default, predict on the best model
            trainer.predict_on_best(testset)
        elif checkpoint_path == 'last':
            # predict on the last model
            trainer.predict_on_last(testset)
        else:
            trainer.predict(testset)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multimodal classification'
    )

    parser.add_argument('--exp_name', type=str, metavar='-n', help="experiment name")
    parser.add_argument('--config', type=str, metavar='-c', help="configuration file path")
    parser.add_argument('--tensorboard_off', action='store_false', help='whether activate tensorboard logger')
    parser.add_argument('--resume', type=str, metavar='-r', help="resume the training from <checkpoint> path")
    parser.add_argument('--train', action='store_true', help="whether train the model")

    args = parser.parse_args()
    configs = read_configs(args.config)
    
    main(args, configs)
    
    