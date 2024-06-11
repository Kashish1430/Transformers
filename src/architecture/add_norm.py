import torch
import math
import torch.nn as nn
from src.utils import get_mean_std

class LayerNormalisation(nn.Module):
    def __init__(self, parameters, eps = 10e-5):
        super().__init__()
        self.parameters = parameters
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.parameters))
        self.beta = nn.Parameter(torch.zeros(self.parameters))
        
    def forward(self, x):
        dims = [-(i+1) for i in range(len(self.parameters))]
        mean, std = get_mean_std(x, dims, self.eps)
        output = (x - mean) / std
        output = ( self.gamma * output ) + self.beta
        return output
    
if __name__ == '__main__':
    inputs = torch.rand((32,100,512))
    layer_norm = LayerNormalisation(inputs.shape[-1:])
    output = layer_norm(inputs)
    print(output.shape)