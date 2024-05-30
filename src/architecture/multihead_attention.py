import torch
import math
import torch.nn as nn
from src.utils import create_temp_input, get_qkv, get_attention, get_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEAD):
        super().__init__()
        self.BATCH = BATCH
        self.SEQ_LENGTH = SEQ_LENGTH
        self.INPUT_DIM = INPUT_DIM
        self.REP = REP
        self.TOTAL_HEADS = TOTAL_HEADS
        self.DIMS_PER_HEAD = DIMS_PER_HEAD
        self.linear_layer = nn.Linear(self.REP, self.REP)
        
    def forward(self, x):
        q, k, v = get_qkv(x, self.INPUT_DIM, self.REP, self.TOTAL_HEADS, self.DIMS_PER_HEAD)
        print(q.shape)
        print(k.shape)
        print(v.shape)
        attention = get_attention(q, k, v)
        attention = self.linear_layer(attention)
        print(attention.shape)
        

if __name__ == '__main__':
    BATCH = 32
    SEQ_LENGTH = 100
    INPUT_DIM = 1024
    REP = 512 # D_MODEL
    TOTAL_HEADS = 8
    DIMS_PER_HEAD = REP // TOTAL_HEADS
    temp_input = create_temp_input(BATCH, SEQ_LENGTH, INPUT_DIM)
    multi_head = MultiHeadAttention(BATCH, SEQ_LENGTH, INPUT_DIM, REP, TOTAL_HEADS, DIMS_PER_HEAD)
    output = multi_head(temp_input)