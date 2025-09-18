import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Scaled Dot-Product Attention module.

    Args:
        None

    Forward Args:
        query (Tensor): Query tensor of shape [B, h, T, d_k]
        key (Tensor): Key tensor of shape [B, h, T, d_k]
        value (Tensor): Value tensor of shape [B, h, T, d_k]
        mask (Tensor, optional): Mask tensor broadcastable to attention scores.
        dropout (callable, optional): Dropout function to apply to attention weights.

    Returns:
        output (Tensor): Output tensor after attention [B, h, T, d_k]
        attn (Tensor): Attention weights [B, h, T, T]
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return torch.matmul(attn, value), attn
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    Args:
        attn_heads (int): Number of attention heads.
        embed_size (int): Embedding dimension.
        dropout (float): Dropout probability.

    Forward Args:
        query (Tensor): Query tensor of shape [B, T, embed_size]
        key (Tensor): Key tensor of shape [B, T, embed_size]
        value (Tensor): Value tensor of shape [B, T, embed_size]
        mask (Tensor, optional): Mask tensor broadcastable to attention scores.

    Returns:
        Tensor: Output tensor after multi-head attention and projection [B, T, embed_size]
    """
    def __init__(self, attn_heads, embed_size, dropout):
        super().__init__()
        assert embed_size % attn_heads == 0, "embed_size must be divisible by attn_heads"
        self.head_size = embed_size // attn_heads
        self.attn_heads = attn_heads
        self.lins = nn.ModuleList([
            nn.Linear(embed_size, embed_size) for _ in range(3)  # W_q, W_k, W_v
        ])
        self.proj = nn.Linear(embed_size, embed_size)  # W_o
        self.attention = Attention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            query (Tensor): [B, T, embed_size]
            key (Tensor): [B, T, embed_size]
            value (Tensor): [B, T, embed_size]
            mask (Tensor, optional): [B, 1, 1, T] or broadcastable

        Returns:
            Tensor: [B, T, embed_size]
        """
        batch_size = query.size(0)
        # Apply linear projections, reshape, and transpose for multi-head
        query, key, value = [
            lin(x)
            .reshape(batch_size, -1, self.attn_heads, self.head_size)
            .transpose(1, 2)
            for lin, x in zip(self.lins, (query, key, value))
        ]
        
        if mask is not None:
            assert mask.dtype == torch.bool, "Use boolean masks (True=allow, False=block)."
            # mask shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # Concatenate heads and project
        x = (
            x.transpose(1, 2)
            .reshape(batch_size, -1, self.attn_heads * self.head_size)
        )
        # Clean up
        del query, key, value
        return self.proj(x) # no dropout