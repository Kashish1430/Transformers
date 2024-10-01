import torch
import torch.nn as nn
from src.architecture.positional_encodings import PositionalEncodings
from src.architecture.multihead_attention import MultiHeadAttention
from src.architecture.add_norm import LayerNormalisation
from src.architecture.MultiCrossHead_attention import MultiCrossHeadAttention
from src.architecture.positionwise_FFN import FeedForwardNN
from src.utils import get_mask

class Decoder_Single_Layer(nn.Module):
    def __init__(self, BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEAD):
        super().__init__()
        self.REP = REP
        self.positional_encodings = PositionalEncodings(SEQ_LENGTH, REP)
        self.multihead_attention = MultiHeadAttention(BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEAD)
        self.layer_norm = LayerNormalisation([REP])
        self.cross_attention = MultiCrossHeadAttention(REP, TOTAL_HEADS)
        self.positionwise_ffn = FeedForwardNN(REP, 4* REP)
        pass
    
    def forward(self, x, y, mask=None):
        positions = self.positional_encodings()
        y += positions
        residuals = y.clone()
        y = self.multihead_attention(y, mask)
        y = self.layer_norm(y + residuals)
        residuals = y.clone()
        attention =  self.cross_attention(x, y)
        attention = self.layer_norm(attention + residuals)
        residuals = attention.clone()
        attention = self.positionwise_ffn(attention)
        attention = self.layer_norm(attention + residuals)
        print(attention.shape)
        return attention
    
if __name__ == '__main__':
    a = torch.randn((32, 100,512)) # This comes from encoding layer.
    b = torch.randn((32, 100,512))
    decoder_sing = Decoder_Single_Layer(32, 100, 512, 512, 8, 64)
    attention = decoder_sing(a,
                             b,
                             get_mask([100, 100])
                             )
    