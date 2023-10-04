from .dataset import DVM
from .tabular_utils import TextEmbedder, OneHotEmbedder, Scarf, RandomMask


def get_transform(transform_configs):
    transform = None
    return transform

def get_dataset(name, split, transform_configs, *args, **kwargs):
    transforms = get_transform(transform_configs)
    if name == 'dvm':
        dataset = DVM(split=split, transforms=transforms, *args, **kwargs)
    else:
        raise ValueError
    
    return dataset, transforms

if __name__ == "__main__":
    pass