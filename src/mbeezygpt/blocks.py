"""
TODO
"""

import torch.nn as nn

from .communication import MultiHeadAttention
from .computation import FeedFoward


class StandardBlock(nn.Module):
    """ Transformer block: communication followed by computation

    Parameters
    ----------
    TODO
    """

    def __init__(self, n_embed, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size, n_embed, dropout)
        self.ffwd = FeedFoward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        """
        TODO
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
