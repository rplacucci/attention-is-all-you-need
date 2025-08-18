import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.lut = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.embed_size)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_size)
        pos = torch.arange(0, max_len).unsqueeze(1)

        # compute 1/L^(2i/embed_size) in log space with L=10,000
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(1e4) / embed_size)) 
        pe[:, 0::2] = torch.sin(pos * div_term) # even positions
        pe[:, 1::2] = torch.cos(pos * div_term) # odd positions
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)    # add residual connection and set const
        return self.dropout(x)