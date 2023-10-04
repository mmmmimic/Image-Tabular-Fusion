from registry import *
import torchvision.transforms as T

def get_criterion(name, *args, **kwargs):
    return LOSS[name](*args, **kwargs)

def build_criterion(crtn_dict):
    wrapped_dict = {}
    for name in crtn_dict.keys():
        wrapped_dict[name+'_loss'] = (crtn_dict[name]['weight'], get_criterion(name=name, **crtn_dict[name]['config']))
    return wrapped_dict

def wrap_model(model, wrapper_type, device):
    return WRAPPER[wrapper_type](model, device)

def build_models(model_name, model_config):
    return MODEL[model_name](**model_config)

def build_embedder(embedder_name, embedder_config):
    return Embedder[embedder_name](**embedder_config)

def get_image_transform(image_shape, aug_rate: float, augs: list, norm=False):
    img_resize = T.Resize(tuple(image_shape))
    
    if norm:
        post_tfs = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
    else:
        post_tfs = T.ToTensor()
    
    if len(augs):
        transform = eval(augs)
        combined_transform = T.RandomChoice([
            img_resize, 
            transform
        ], p=[1-aug_rate, aug_rate])
    else:
        combined_transform = img_resize
    combined_transform = T.Compose(
        [
            combined_transform, 
            post_tfs
        ]
    )
    
    return combined_transform
        
def get_tabular_transform(augs: dict):
    '''
    get transforms for tabular data from a dictionary
    e.g., 
    {
        name: scarf
        config:
            corrupt_rate: 0.7
    }
    '''
    name = augs['name']
    config = augs['config']
    return TABULAR_TRANSFORM[name](**config)
    
def get_transform(transform_configs, split):
    tfs = {}
    if split == 'train':
        if 'tabular' in transform_configs.keys():
            tfs['tab_tf'] = get_tabular_transform(transform_configs['tabular'])
        if 'image' in transform_configs.keys():
            image_tf_configs = transform_configs['image']
            tfs['img_tf'] = get_image_transform(image_tf_configs['train_shape'], image_tf_configs['aug_rate'], image_tf_configs['train_augs'], image_tf_configs['norm'])
    elif split == 'test':
        if 'tabular' in transform_configs.keys():
            tfs['tab_tf'] = lambda x, y: y
        if 'image' in transform_configs.keys():
            image_tf_configs = transform_configs['image']
            tfs['img_tf'] = get_image_transform(image_tf_configs['test_shape'], 0, image_tf_configs['test_augs'], image_tf_configs['norm'])     
    return tfs
  
def build_dataset(name, config):
    # convert transforms and tab_embedder (if applicable)
    if 'tab_embedder' in config.keys():
        tab_embedder = build_embedder(config['tab_embedder']['name'], config['tab_embedder']['config'])
        config['tab_embedder'] = tab_embedder
        
    train_transforms = get_transform(config['transforms'], split='train')
    test_transforms = get_transform(config['transforms'], split='test')
    
    # get train, val and test set respectively
    config['transforms'] = train_transforms
    trainset = DATASET[name](split='train', **config)
    config['transforms'] = test_transforms
    valset = DATASET[name](split='val', **config)
    testset = DATASET[name](split='test', **config)
    
    return trainset, valset, testset

if __name__  == "__main__":
    import matplotlib.pyplot as plt
    data_config = {
        'dataset': 'dvm',
        'numerical': True,
        'cache_images': False,
        'tab_embedder':
            {
                'name': 'onehot',
                'config': {}
             },
        'transforms':
            {
                'tabular': 
                    {
                        'name': 'scarf',
                        'config': {'corrupt_rate': 0.7}
                    },
                'image':
                    {
                        'train_shape': [128, 128],
                        'test_shape': [128, 128],
                        'norm': False, 
                        'aug_rate': 0.95,
                        'train_augs': """T.Compose([T.RandomApply([T.ColorJitter(brightness=[0.2, 1.8],
                            contrast=[0.2, 1.8], saturation=[0.2, 1.8], hue=0)], p=0.8), 
                            T.RandomGrayscale(p=0.2), T.RandomApply([T.GaussianBlur(kernel_size=(29, 29), sigma=(0.1, 2.0))], p=0.5), 
                            T.RandomResizedCrop(size=(128, 128), scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
                            T.RandomHorizontalFlip(p=0.5)])""",
                        'test_augs': ""
                    }
            }
    }

    trainset, valset, testset = build_dataset(data_config['dataset'], data_config)
        
    data = trainset[0]
    image = data['image']
    tab_line = data['tab_line']
    label = data['label']
    print(tab_line, tab_line.shape, label, image.shape)
    print(image.min(), image.max())
    plt.figure()
    plt.imshow(image.permute(1,2,0).numpy())
    plt.show()