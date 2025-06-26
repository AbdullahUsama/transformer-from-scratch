import torch 
import torch.nn as nn
import math
    
class LayerNormalization(nn.Module):
    
    def __init__(self, eps:float = 10**-6)->None:
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        normalized_x = (x - mean) / (std + self.eps)
        return self.alpha * normalized_x + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout:float)->None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) #w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # #w2 and b2

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x