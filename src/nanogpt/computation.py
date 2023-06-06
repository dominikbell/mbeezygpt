"""
TODO
"""

import torch.nn as nn

from .utils import new_gelu


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity

    Parameters
    ----------
    TODO
    """

    def __init__(self, n_embed, dropout, ratio=10):
        super().__init__()
        self.proj_up = nn.Linear(n_embed, ratio * n_embed)
        self.activ = nn.ReLU()
        self.proj_down = nn.Linear(ratio * n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        TODO
        """
        x = self.proj_up(x)
        x = self.activ(x) # ReLu
        # x = new_gelu(x)
        x = self.proj_down(x)
        x = self.dropout(x)
        return x
