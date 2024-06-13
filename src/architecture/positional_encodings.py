import torch
import torch.nn as nn
from src.utils import get_denominator, get_position_encodings

class PositionalEncodings(nn.Module):
    def __init__(self, SEQ_LENGTH, REP):
        super().__init__()
        self.SEQ_LENGTH = SEQ_LENGTH
        self.REP = REP
        
    def forward(self):
        denominator = get_denominator(self.REP)
        flattened = get_position_encodings(self.SEQ_LENGTH, denominator)
        return flattened
        
if __name__ == '__main__':
    BATCH = 32
    SEQ_LENGTH = 100
    INPUT_DIM = 1024
    REP = 512 # D_MODEL
    TOTAL_HEADS = 8
    DIMS_PER_HEAD = REP // TOTAL_HEADS
    
    posi = PositionalEncodings(SEQ_LENGTH, REP)
    output = posi.forward()
    print(output)
    print(output.shape)
    