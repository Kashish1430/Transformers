import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, REP, hidden_layer):
        super().__init__()
        self.REP = REP
        self.hidden_layer = hidden_layer
        self.drop_prob = 0.1
        self.linear_layer_1 = nn.Linear(self.REP, self.hidden_layer)
        self.linear_layer_2 = nn.Linear(self.hidden_layer, self.REP)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_prob)
        
    def forward(self, x):
        x = self.linear_layer_1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.linear_layer_2(x)
        return x

if __name__  == '__main__':
    inputs = torch.randn((32, 100, 512))
    ffn = FeedForwardNN(512,  4*512)
    output = ffn(inputs)
    print(output.shape)
    

        
        