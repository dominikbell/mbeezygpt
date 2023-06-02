"""
TODO
"""

import torch.nn as nn


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity

    Parameters
    ----------
    TODO
    """

    def __init__(self, n_embd, dropout, ratio=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, ratio * n_embd),
            nn.ReLU(),
            nn.Linear(ratio * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        TODO
        """
        return self.net(x)
