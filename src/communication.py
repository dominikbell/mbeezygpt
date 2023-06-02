"""
TODO
"""

import torch
from torch.nn import functional as F
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    """ one head of self-attention

    Parameters
    ----------
    TODO
    """

    def __init__(self, head_size, block_size, n_embed, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        TODO
        """
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5

        # since we are decoding, tokens only talk to the other tokens in the past
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel

    Parameters
    ----------
    TODO
    """

    def __init__(self, num_heads, head_size, block_size, n_embed, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleHeadAttention(head_size, block_size, n_embed, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        TODO
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
