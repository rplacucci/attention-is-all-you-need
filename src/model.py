import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from embeddings import TokenEmbedding, PositionalEncoding

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
    def __init__(self, embed_size, dropout, prenorm=False):
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.prenorm = prenorm
    
    def forward(self, x, sublayer):
        if self.prenorm:
            return x + self.dropout(sublayer(self.norm(x)))     # pre-norm
        else:
            return self.norm(x + self.dropout(sublayer(x)))     # post-norm

class EncoderLayer(nn.Module):
    """
    Transformer encoder layer.
    """
    def __init__(self, attn_heads, embed_size, ff_size, dropout=0.1):
        """
        Args:
            attn_heads (int): Number of attention heads.
            embed_size (int): Embedding dimension.
            ff_size (int): Feedforward layer dimension.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.attention = MultiHeadAttention(attn_heads, embed_size, dropout)
        self.feed_forward = PositionWiseFeedForward(embed_size, ff_size, dropout)
        self.sublayers = nn.ModuleList([SublayerConnection(embed_size, dropout) for _ in range(2)])

    def forward(self, x, mask):
        """
        Args:
            x (Tensor): Input tensor.
            mask (Tensor): Attention mask.
        Returns:
            Tensor: Output tensor.
        """
        x = self.sublayers[0](x, lambda x: self.attention(x, x, x, mask))
        x = self.sublayers[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_layers, attn_heads, embed_size, ff_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(attn_heads, embed_size, ff_size, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, attn_heads, embed_size, ff_size, dropout=0.1):
        super().__init__()
        self.s_attention = MultiHeadAttention(attn_heads, embed_size, dropout)
        self.x_attention = MultiHeadAttention(attn_heads, embed_size, dropout)
        self.feed_forward = PositionWiseFeedForward(embed_size, ff_size, dropout)
        self.sublayers = nn.ModuleList([SublayerConnection(embed_size, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayers[0](x, lambda x: self.s_attention(x, x, x, tgt_mask))   # self-attn
        x = self.sublayers[1](x, lambda x: self.x_attention(x, m, m, src_mask))   # cross-attn
        x = self.sublayers[2](x, self.feed_forward)  # FFN
        return x
    
class Decoder(nn.Module):
    def __init__(self, n_layers, attn_heads, embed_size, ff_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(attn_heads, embed_size, ff_size, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class Generator(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super().__init__()
        self.proj = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)  # log-softmax + KL div
    
class Transformer(nn.Module):
    def __init__(self, n_layers, max_len, vocab_size, embed_size, ff_size, attn_heads, dropout):
        super().__init__()
        self.tok_embed = TokenEmbedding(vocab_size, embed_size)
        self.pos_embed = PositionalEncoding(embed_size, max_len, dropout)
        self.encoder = Encoder(n_layers, attn_heads, embed_size, ff_size, dropout)
        self.decoder = Decoder(n_layers, attn_heads, embed_size, ff_size, dropout)
        self.generator = Generator(embed_size, vocab_size)

        # init weights
        self.apply(self._init_weights)

        # tie weights in generator and embedding
        self.generator.proj.weight = self.tok_embed.lut.weight

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoded = self.encoder(src, src_mask)
        decoded = self.decoder(encoded, src_mask, tgt, tgt_mask)
        return self.generator(decoded)
    
    def embed(self, x):
        return self.pos_embed(self.tok_embed(x))

    def encode(self, src, src_mask):
        return self.encoder(self.embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.embed(tgt), memory, src_mask, tgt_mask)
    
    @staticmethod
    def _init_weights(module: nn.Module):
        """
        Original Transformer-style init:
          - Weights ~ U(-0.1, 0.1)
          - Biases = 0
          - LayerNorm: weight=1.0, bias=0.0
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.uniform_(module.weight, -0.1, 0.1)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)