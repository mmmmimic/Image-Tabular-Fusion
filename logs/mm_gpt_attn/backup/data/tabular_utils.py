import numpy as np
import pandas as pd
import random
import clip
import torch
from transformers import RobertaTokenizer

def onehot(tab_value: int, field_length: int) -> np.ndarray:
    # convert a number (category index) to its onehot embedding
    onehot_array = np.zeros(field_length, dtype=np.float32)
    onehot_array[tab_value] = 1
    # onehot_array = torch.nn.functional.one_hot(torch.tensor(tab_value), num_classes=field_length).numpy() # very slow, discarded
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
    def __init__(self, cellwise=True, model='clip', context=None, shuffle=False, mask_rate=0, chatgpt_tmpl=None) -> None:
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
      self.context = context
      self.shuffle = shuffle
      self.mask_rate = mask_rate
      if chatgpt_tmpl is not None:
        with open(chatgpt_tmpl, 'r') as f:
          content = f.read()
        self.chatgpt_tmpl = content
      else:
        self.chatgpt_tmpl = None
          
    def get_line(self, df: pd.DataFrame, meta_info: dict, index: int) -> np.ndarray:
        '''
        args
            df: a pandas dataframe
            meta: the meta information of the dataframe (column names, field length)
            index: line index
        '''
        columns = filter(lambda x: meta_info[x]['type'] in ['continuous', 'categorical'], meta_info.keys())
        line = {}
        
        if self.chatgpt_tmpl != None:
          sentence = self.chatgpt_tmpl
          # use chatgpt tmpl:
          for c in columns:
            t = '<' + meta_info[c]['full_name'] + '>'
            sentence = sentence.replace(t, str(df[c].values[index]))
            line_sentence = sentence.split('\n')

        else:   
          line_sentence = []          
          for c in columns:
            if self.context is None:
              cell_sentence = f"{meta_info[c]['full_name']}: "
            else:
              cell_sentence = f"The {meta_info[c]['full_name']} of the {self.context} is "
              
            if not self.mask_rate:
              cell_sentence = cell_sentence + str(df[c].values[index])
            else:
              if np.random.rand() < self.mask_rate:
                cell_sentence = cell_sentence + 'missing'
              else:
                cell_sentence = cell_sentence + str(df[c].values[index])
                
            line_sentence.append(cell_sentence)
          if not self.cellwise:
            # combine cell contents into one sentence
            if self.shuffle:
              np.random.shuffle(line_sentence)
      
            line_sentence = ', '.join(line_sentence)
            line_sentence = [line_sentence]

        line['line_sentence'] = line_sentence
        
        # generate text embedding
        if self.model == 'clip':
          # clip
          line_embd = self.tokenizer(line_sentence, truncate=True)
          line['line_embd'] = line_embd
          
        else:
          # roberta
          line_embd = self.tokenizer(line_sentence[0], return_tensors='pt', padding=True)
          input_ids = line_embd['input_ids']
          attention_mask = line_embd['attention_mask']
                        
          if input_ids.size(1) < 120:
            gap = 120 - input_ids.size(1)
            input_ids = torch.cat((input_ids, torch.zeros_like(input_ids)[...,:gap]), dim=1)
            attention_mask = torch.cat((attention_mask, torch.zeros_like(attention_mask)[...,:gap]), dim=1)
          elif input_ids.size(1) > 120:
            input_ids = input_ids[...,:120]
            attention_mask = attention_mask[...,:120]
        
          line_embd['input_ids'] = input_ids
          line_embd['attention_mask'] = attention_mask
            
          line['line_embd'] = line_embd

        return line
   
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
        subject[i] = 'missing'
        
      return subject