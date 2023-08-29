from typing import Any
import numpy as np
import pandas as pd
import torch.nn.functional as F
import random

def onehot(tab_value: int, field_length: int) -> np.ndarray:
    # convert a number (category index) to its onehot embedding
    onehot_array = np.zeros(field_length, dtype=np.float32)
    onehot_array[tab_value] = 1
    return onehot_array

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
        line_embd = np.array(line_embd)
        return line_embd        
    
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

class WordEmbedder:
    def __init__(self, bert) -> None:
        pass
        # load pretrained models here
                
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
    
    
class Scarf:
    def __init__(self, corrupt_rate=0.7) -> None:
       self.c = corrupt_rate
       
    def __call__(self, index, tab_data: pd.DataFrame):
      # output should be another pandas dataframe
      self.marginal_distributions = tab_data.transpose().values.tolist()
      corrupt_tab_data = tab_data.copy()
      corrupt_tab_data.iloc[index,:] = self.corrupt(tab_data.iloc[index,:])
      return corrupt_tab_data

    def corrupt(self, subject):
      """
      Creates a copy of a subject, selects the indices 
      to be corrupted (determined by hyperparam corruption_rate)
      and replaces their values with ones sampled from marginal distribution
      """
      subject = subject.copy()

      indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
      
      for i in indices:
        subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
      return subject