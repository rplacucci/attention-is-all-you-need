import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward sublayer used in transformers.

    Args:
        embed_size (int): Dimensionality of input and output.
        ff_size (int): Dimensionality of the inner feed-forward layer.
        dropout (float): Dropout probability.

    Forward:
        x (Tensor): Input tensor of shape (batch_size, seq_len, embed_size).
        Returns:
            Tensor of shape (batch_size, seq_len, embed_size).
    """
    def __init__(self, embed_size, ff_size, dropout):
        super().__init__()
        self.lin1 = nn.Linear(embed_size, ff_size)
        self.lin2 = nn.Linear(ff_size, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(self.relu(self.lin1(x))))

class SublayerConnection(nn.Module):
    """
    Implements layer normalisation followed by dropout and a residual connection (pre-norm).

    Args:
        embed_size (int): Dimensionality of input and output.
        dropout (float): Dropout probability.

    Forward:
        x (Tensor): Input tensor.
        sublayer (Callable): Sublayer function to apply to x.
        Returns:
            Tensor after applying sublayer, residual connection, and normalization.
    """
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))   # pre-norm with dropout