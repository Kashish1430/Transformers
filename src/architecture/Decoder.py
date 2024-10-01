import torch
import torch.nn as nn
from src.architecture.Decoder_Single_Layer import Decoder_Single_Layer
from src.utils import get_mask



class Sequential(nn.Sequential):
    def __init_(self):
        super().__init__()
    
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)
        return y 

class Decoder(nn.Module):
    def __init__(self, BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEAD, num_layers=5):
        super().__init__()
        self.decoder_layers = Sequential(*[Decoder_Single_Layer(BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEAD)
                                           for _ in range(num_layers)])
        
    def forward(self, x, y, mask):
        y = self.decoder_layers(x, y, mask)
        return y

if __name__ == '__main__':
    x = torch.randn((32, 100, 512))
    y = torch.randn((32, 100, 512))
    decoder = Decoder(32, 100, 512, 512, 8, 64)
    outputs = decoder(x, y, get_mask([100, 100]))
    
    