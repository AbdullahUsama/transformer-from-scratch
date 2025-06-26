import torch 
import torch.nn as nn
import math
    
class PositionalEncoding(nn.Module):

    def __inint__(self, d_model:int, seq_len:int, dropout:float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model  
        self.seq_len = seq_len

        #create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector shape of (seq_len,1)
        position = torch.arrange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #apply the sin to even 
        pe[:, 0::2] = torch.sin(position * div_term)
        #apply the cos to odd
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # add batch dimension

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)