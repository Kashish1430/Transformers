import torch
import torch.nn as nn
from src.utils import get_attention
from src.utils import get_kv, get_q, get_mask

class MultiCrossHeadAttention(nn.Module):
    def __init__(self, REP, total_heads):
        super().__init__()
        self.total_heads = total_heads
        self.dims_per_head = REP//total_heads
        self.linear_layer = nn.Linear(REP, REP)
        
    def forward(self, x, y, mask=None):
        batch, seq, rep = x.size()
        k, v = get_kv(x, rep, rep, self.total_heads, self.dims_per_head)
        print(k.shape)
        print(v.shape)
        q = get_q(y, rep, rep, self.total_heads, self.dims_per_head)
        print(q.shape)
        values = get_attention(q, k, v, mask)
        print(values.shape)
        output = self.linear_layer(values)
        print(output.shape)
        return output
        
if __name__ == '__main__':
    temp_input_enc = torch.randn((32, 100, 512))
    temp_input_y = torch.randn((32, 100, 512))
    cross_head= MultiCrossHeadAttention(512, 8)
    output = cross_head(temp_input_enc, temp_input_y )
    
    