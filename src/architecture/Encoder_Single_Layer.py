import torch
import torch.nn as nn
from src.architecture.positional_encodings import PositionalEncodings
from src.architecture.multihead_attention import MultiHeadAttention
from src.architecture.add_norm import LayerNormalisation
from src.architecture.positionwise_FFN import FeedForwardNN

class EncoderLayer(nn.Module):
    def __init__(self, BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEAD, HIDDEN_LAYER):
        super().__init__()
        self.positional_encodings = PositionalEncodings(SEQ_LENGTH, REP)
        self.multihead_attention = MultiHeadAttention(BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEAD)
        self.layer_norm = LayerNormalisation([REP])
        self.positionwise_ffn = FeedForwardNN(REP, HIDDEN_LAYER)
    
    def forward(self, x):
        print('Before Positional Encodings: ',x.shape)
        positions = self.positional_encodings()
        x += positions
        print('After Position Encodings: ',x.shape)
        residuals = x
        x = self.multihead_attention(x)
        print('After multihead attention: ',x.shape)
        x = self.layer_norm(x + residuals)
        residuals = x
        print('After 1st layer normalisation:', x.shape)
        x = self.positionwise_ffn(x)
        print('After positionwise ffn: ',x.shape)
        x = self.layer_norm(x + residuals)
        print('After 2nd Layer norm: ',x.shape)
        return x
    
if __name__ == '__main__':
    enc_layer = EncoderLayer(32, 100, 512, 512, 8, 512//8, 4*512)
    inputs = torch.randn((32,100,512))
    output = enc_layer(inputs)