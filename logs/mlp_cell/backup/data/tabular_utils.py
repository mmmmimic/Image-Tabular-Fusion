import numpy as np
import pandas as pd
import random
import clip
import torch
import regex as re
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from transformers import RobertaTokenizer

def onehot(tab_value: int, field_length: int) -> np.ndarray:
    # convert a number (category index) to its onehot embedding
    onehot_array = np.zeros(field_length, dtype=np.float32)
    onehot_array[tab_value] = 1
    return onehot_array

class DefaultEmbedder:
  def __init__(self):
    pass
  
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
      line_embd = torch.tensor(line_embd)
      return {
                'line_embd': line_embd.float()
                }       
    
class OneHotEmbedder(DefaultEmbedder):
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
        line_embd = torch.tensor(line_embd)
        return {
                'line_embd': line_embd.float()
                }

class TextEmbedder(DefaultEmbedder):
    def __init__(self, cellwise=True, model='clip', withhead=True, word_limit=77) -> None:
      '''
      Encode tabular content into word embeddings with a pretrained LLM
      args:
        cellwise (bool): if True, tokenize each cell content respectively, 
        otherwise tokenize the full tabular information will be represented as a sentence
        model (str): language model type
        withhead (bool): if the cell sentence includes the table head 
        word_limit (int): the maximum number of words in each sentence. If a sentence is too long, its words will be randomly dropped
      '''
      self.cellwise = cellwise
      
      if model == 'clip':
        self.tokenizer = clip.tokenize
      elif model == 'roberta':
        self.tokenizer = RobertaTokenizer.from_pretrained('Roberta-base')
      else:
        raise NameError
      
      self.model = model
      self.withhead = withhead
      self.word_limit = word_limit
          
    def get_line(self, df: pd.DataFrame, meta_info: dict, index: int) -> np.ndarray:
        '''
        args
            df: a pandas dataframe
            meta: the meta information of the dataframe (column names, field length)
            index: line index
        '''
        columns = filter(lambda x: meta_info[x]['type'] in ['continuous', 'categorical'], meta_info.keys())
        _tokenizer = _Tokenizer()
        
        line_sentence = []          
        for c in columns:
          if self.withhead:
            cell_sentence = f"{meta_info[c]['full_name']}: "
          else:
            cell_sentence = ''
          cell_sentence = cell_sentence + str(df[c].values[index])
          
          # if self.model == 'clip':
          #   if len(_tokenizer.encode(cell_sentence)) >= self.word_limit:
          #     # cutoff
          #     cell_sentence = ' '.join(cell_sentence.split[' '][:self.word_limit])
          line_sentence.append(cell_sentence)
        
        if not self.cellwise:
          # combine cell contents into one sentence
          line_sentence = ', '.join(line_sentence)
          # if self.model == 'clip':
          #   while len(_tokenizer.encode(line_sentence)) >= self.word_limit:
          #     short_sentence = line_sentence.split(', ')
          #     drop_id = np.random.randint(0, len(short_sentence))
          #     short_sentence.pop(drop_id)
          #     line_sentence = ', '.join(short_sentence)
          line_sentence = [line_sentence]
        line_embd = self.tokenizer(line_sentence, truncate=True)
          
        return {
                'line_embd': line_embd,
                'line_sentence': line_sentence
                }
   
class Scarf:
    def __init__(self, corrupt_rate=0.7) -> None:
      '''
      Scarf augmentation
      reference: 
      https://arxiv.org/pdf/2106.15147.pdf 
      The key idea is to replace a fraction of features with samples from their marginal distribution, which is built on the full training data
      '''
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
    
class RandomMask: 
    '''
    Randomly drop a fraction of tabular features with <mask>. Only applicable to tabular word embeddings.  
    '''
    def __init__(self, corrupt_rate=0.7) -> None:
      '''
      Scarf augmentation
      reference: 
      https://arxiv.org/pdf/2106.15147.pdf 
      The key idea is to replace a fraction of features with samples from their marginal distribution, which is built on the full training data
      '''
      self.c = corrupt_rate
    
    def __call__(self, index, tab_data: pd.DataFrame):
      # output should be another pandas dataframe
      corrupt_tab_data = tab_data.copy()
      corrupt_tab_data.iloc[index,:] = self.corrupt(tab_data.iloc[index,:])
      return corrupt_tab_data

    def corrupt(self, subject):
      """
      Creates a copy of a subject, selects the indices 
      to be corrupted (determined by hyperparam corruption_rate)
      and replaces their values with <mask>
      """
      subject = subject.copy()

      indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
      
      for i in indices:
        subject[i] = '<mask>'
        
      return subject