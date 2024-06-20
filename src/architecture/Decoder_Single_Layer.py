import torch
import torch.nn as nn
from src.architecture.positional_encodings import PositionalEncodings
from src.architecture.multihead_attention import MultiHeadAttention
from src.architecture.add_norm import LayerNormalisation

class Decoder_Single_Layer(nn.Module):
    def __init__(self, BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEAD):
        super().__init__()
        self.positional_encodings = PositionalEncodings(SEQ_LENGTH, REP)
        self.multihead_attention = MultiHeadAttention(BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEAD)
        self.layer_norm = LayerNormalisation([REP])
        pass