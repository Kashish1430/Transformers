import torch
import math

def get_denominator(REP):
    half_position = torch.arange(0, REP, 2).float()
    denominator = torch.pow(10000, half_position/REP)
    return denominator


def get_position_encodings(SEQ_LENGTH, denominator):
    positions = torch.arange(0, SEQ_LENGTH).float().reshape(SEQ_LENGTH, 1)
    even_pe = torch.sin(positions/denominator)
    odd_pe = torch.sin(positions/denominator)
    
    stacked = torch.stack([even_pe, odd_pe], dim = 2)
    flattened = torch.flatten(stacked, start_dim=1, end_dim = 2)
    return flattened