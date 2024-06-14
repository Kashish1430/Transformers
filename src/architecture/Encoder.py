import torch
import torch.nn as nn
from src.architecture.Encoder_Single_Layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEADS, HIDDEN_LAYER, Nx):
        super().__init__()
        self.encoder_layer = nn.Sequential(*[EncoderLayer(BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEADS, HIDDEN_LAYER) for _ in range(Nx)])
    
    def forward(self, x):
        x = self.encoder_layer(x)
        return x
    
if __name__ == '__main__':
    encoder_obj = Encoder(32, 100, 512, 512, 8, 512//8, 4*512, 5)
    inputs = torch.randn((32,100,512))
    outputs = encoder_obj(inputs)
    print(outputs.shape)