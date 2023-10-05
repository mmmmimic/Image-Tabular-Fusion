import torch.nn as nn
import torch
from typing import Any

class ModelWrapper(nn.Module):
    """
    The wrapper wraps an arbitary deep neural network to the format fitting the trainer
    args:
        model (torch.nn.Module)
        device (torch.device)
    """
    def __init__(self, model, device) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.model.to(device)
    
    def forward(self, data) -> Any:
        # input data can be a tensor, a list, or a dictionary
        data = self._to_device(data)
        outputs = self._forward(data)
        return outputs
        
    def _forward(self, data):
        return self.model(data)
        
    def _to_device(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, dict):
            for name in x.keys():
                if isinstance(x[name], torch.Tensor):
                    x[name] = x[name].to(self.device)
            return x
        elif isinstance(x, list):
            for i in range(len(x)):
                if isinstance(x[i], torch.Tensor):
                    x[i] = x[i].to(self.device)
            return x
        else:
            raise TypeError

    def __repr__(self) -> str:
        return(str(self.model))

class DictWrapper(ModelWrapper):
    def __init__(self, model, device) -> None:
        super().__init__(model, device)

    def _forward(self, data):
        # input data can be a tensor, a list, or a dictionary
        data = self._to_device(data)
        outputs = self._forward(data)
        return outputs  
    
    def _forward(self, data):
        # type check
        assert isinstance(data, dict)
        return self.model(**data)    

class TupleWrapper(ModelWrapper):
    def __init__(self, model, device) -> None:
        super().__init__(model, device)
    
    def forward(self, data):
        data = self._to_device(data)
        logit = self._forward(data)
        image, label = data
        return {'image': image, 'label': label, 'logit': logit}
    
    def _forward(self, data):
        # type check
        assert isinstance(data, list) and (len(data)==2) # <image, label>
        return self.model(data[0])  
            
if __name__ == "__main__":
    net = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Conv2d(16, 10, 3)
    )
    model = TupleWrapper(net, device='cpu')
    print(model)
    image, label = torch.rand(5, 3, 224, 224), torch.rand(5)
    print(model([image, label]))