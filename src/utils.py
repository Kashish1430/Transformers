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

def create_temp_input(BATCH, SEQ_LENGTH, INPUT_DIM):
    temp_input = torch.randn((BATCH, SEQ_LENGTH, INPUT_DIM)) #(32, 100, 1024)
    return temp_input

def get_qkv(temp_input, input_dim, REP, total_heads, dim_per_head): 
    qkv_linear_layer = torch.nn.Linear(input_dim, 3 * REP) #(1024, 1536)
    qkv = qkv_linear_layer(temp_input) # (32, 100, 1536)
    qkv = qkv.reshape(temp_input.shape[0], total_heads, temp_input.shape[1], dim_per_head * 3) #(32, 8, 100, 192) 
    q, k, v = qkv.chunk(3, dim=-1)
    return q, k, v

def get_mask(scaled):
    mask = torch.full(scaled.shape, float('-inf'))
    mask = torch.triv(mask, diagonal=1)
    return mask
    
def get_attention(q, k, v, mask=None):
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask != None:
        scaled += mask
    soft = torch.nn.functional.softmax(scaled)
    values = torch.matmul(soft, v)
    values = values.reshape(values.shape[0], values.shape[2], values.shape[1] * values.shape[-1])
    return values
    
def get_mean_std(x, dims, eps):
    mean = x.mean(dim=dims, keepdim=True)
    variance = ((x - mean)**2).mean(dim=dims, keepdim=True)
    std = (variance+eps).sqrt()
    return mean, std
