import numpy as np
import pandas as pd
import torch.nn.functional as F

class DefaultEmbedder:
    def get_line(self, df: pd.DataFrame, meta_info: dict, index: int) -> np.ndarray:
        '''
        args
            df: a pandas dataframe
            meta: the meta information of the dataframe (column names, field length)
            index: line index
        '''
        columns = filter(lambda x: meta_info[x]['type'] in ['continuous', 'categorical'], meta_info.keys())
        line_embd = []
        for c in columns:
            line_embd.append(df[c].values[index])

        line_embd = np.concatenate(line_embd)
        return line_embd        
    
def onehot(tab_value: int, field_length: int) -> np.ndarray:
    # convert a number (category index) to its onehot embedding
    onehot_array = np.zeros(field_length, dtype=np.float32)
    onehot_array[tab_value] = 1
    return onehot_array

class OneHotEmbedder:
    def get_line(self, df: pd.DataFrame, meta_info: dict, index: int) -> np.ndarray:
        '''
        args
            df: a pandas dataframe
            meta: the meta information of the dataframe (column names, field length)
            index: line index
        '''
        columns = filter(lambda x: meta_info[x]['type'] in ['continuous', 'categorical'], meta_info.keys())
        line_embd = []
        for c in columns:
            if meta_info[c]['type'] == 'categorical':
                line_embd.extend(onehot(int(df[c].values[index]), meta_info[c]['field_length']))
            else:
                # continuous values
                line_embd.extend([df[c].values[index]])

        line_embd = np.array(line_embd)
        return line_embd
