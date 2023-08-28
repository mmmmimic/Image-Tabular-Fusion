import numpy as np
import pandas as pd
import torch.nn.functional as F

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

        line_embd = np.concatenate(line_embd)
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
    def __init__(self, tab_data, ) -> None:
        pass
    
    # Tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.eval_one_hot = eval_one_hot
    
    # Classifier
    self.labels = torch.load(labels_path)

    self.train = train

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(torch.clamp(subject[i],min=0,max=self.field_lengths_tabular[i]-1).long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    im = self.data_imaging[index]
    if self.live_loading:
      im = read_image(im)
      im = im / 255

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(im)
    else:
      im = self.default_transform(im)

    if self.eval_one_hot:
      tab = self.one_hot_encode(torch.tensor(self.data_tabular[index]))
    else:
      tab = torch.tensor(self.data_tabular[index], dtype=torch.float)

    label = torch.tensor(self.labels[index], dtype=torch.long)

    return (im, tab), label

  def __len__(self) -> int:
    return len(self.data_tabular)