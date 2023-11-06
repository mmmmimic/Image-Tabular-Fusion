import torch.nn as nn
import torch
from typing import Any

class ModelWrapper(nn.Module):
    """
    The wrapper wraps an arbitary deep neural network to the format fitting the trainer
    args:
        model (torch.nn.Module)
        device (torch.device)
        
    /** Wrapper workflow
    ** x can be an image or a tabular line 
    // Wrapper input (Dataloader item): (tuple) <x, label> or (dict)
    // Model input: (tensor) or (dict)
    // Model output: (tensor) or (dict), if (tensor), the output is 'logit', if (dict), the output must include the key-value pair 'logit'
    // Wrapper output: (dict), i.e., {'logit': logit, 'label': label, ...}
    **/
    """
    def __init__(self, model, wrapper_config, device) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.config = wrapper_config
        self._parse_config()
        
        self.model.to(device)
    
    def _parse_config(self):
        self.wrapper_input_type = self.config['wrapper_input']
        self.model_input_type = self.config['model_input']
        self.model_output_type = self.config['model_output']
        self.kwd = self.config['kwd']
    
    def _format_input(self, data):
        # wrapper input -> model input
        if self.model_input_type == 'tensor':
            if self.wrapper_input_type == 'tuple':
                model_input = data[0]
            elif self.wrapper_input_type == 'dict':
                model_input = data[self.kwd]
                
        elif self.model_input_type == 'dict':
            if self.wrapper_input_type == 'tuple':
                model_input = {}
                model_input[self.kwd] = data[0]
                model_input['label'] = data[1]
            else:
                model_input = data
        
        return model_input
    
    def _format_output(self, model_output, wrapper_input):
        # model output -> wrapper output
        if self.model_output_type == 'tensor':
            if self.wrapper_input_type == 'dict':
                wrapper_output = wrapper_input
                wrapper_output['logit'] = model_output
            elif self.wrapper_input_type == 'tuple':
                wrapper_output = {'logit': model_output, 'label': wrapper_input[1]}
        elif self.model_output_type == 'dict':
            wrapper_output = model_output
            wrapper_output['label'] = wrapper_input['label']
        
        return wrapper_output
    
    def forward(self, wrapper_input) -> Any:
        # input data can be a tensor, a list, or a dictionary
        wrapper_input = self._to_device(wrapper_input)
        model_input = self._format_input(wrapper_input)
        model_output = self._forward(model_input)
        wrapper_output = self._format_output(model_output, wrapper_input)
        return wrapper_output
        
    def _forward(self, data):
        if self.model_input_type == 'dict':
            return self.model(**data)
        elif self.model_input_type == 'tensor':
            return self.model(data)
        else:
            raise TypeError(f"Data type {type(data)} is not supported.")
        
    def _to_device(self, x):
        if self.wrapper_input_type == 'dict':
            for name in x.keys():
                if isinstance(x[name], torch.Tensor):
                    x[name] = x[name].to(self.device)
            return x
        elif self.wrapper_input_type == 'tuple':
            x[0] = x[0].to(self.device)
            x[1] = x[1].to(self.device)
            return x
        else:
            raise TypeError(f"Data type {type(x)} is not supported.")

    def __repr__(self) -> str:
        return(str(self.model))
            
if __name__ == "__main__":
    image, label = torch.rand(5, 3, 224, 224), torch.rand(5)
    
    # case 1
    config = {
            'wrapper_input': 'tuple',
            'model_input': 'tensor',
            'model_output': 'tensor',
            'kwd': 'image'
    }
    net = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Conv2d(16, 10, 3)
    )
    model = ModelWrapper(net, config, device='cpu')
    print(model)
    print(model([image, label]))
    
    # case 2
    config = {
            'wrapper_input': 'dict',
            'model_input': 'tensor',
            'model_output': 'tensor',
            'kwd': 'image'
    }
    model = ModelWrapper(net, config, device='cpu')
    print(model)
    print(model({'image': image, 'label': label}))  
    
    