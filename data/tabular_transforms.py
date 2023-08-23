import numpy as np
import pandas as pd
import torch.nn.functional as F

def onehot(tab_value: int, tab_meta):
    return F.one_hot()