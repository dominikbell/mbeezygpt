"""
TODO
"""

import torch  # we use PyTorch: https://pytorch.org
from torch.nn import functional as F
import torch.nn as nn

from blocks import StandardBlock


class BigramLanguageModel(nn.Module):
    """
    Parameters
    ----------
    vocab_size: int
        the number of tokens

    block_size: int
        the size of each block

    n_embed: int
        the embedding dimensionality

    n_heads: int
        the number of single attention heads

    n_layers: int
        the number of blocks layered on top of each other

    dropout: float
        the dropout ratio (< 1.0)
    """

    def __init__(self, vocab_size, block_size, n_embed, n_heads, n_layers, dropout):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # add blocks
        self.blocks = nn.Sequential(
            *[StandardBlock(n_embed, n_heads, block_size, dropout) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, inputs, targets=None):
        """
        TODO

        Parameters
        ----------
        inputs : tensor (?)
            (B,T) tensor

        targets : tensor, optional
            (B, T) tensor
        """
        B, T = inputs.shape
        # idx and targets are both (B,T) tensor of integers
        tok_embed = self.token_embedding_table(inputs)  # (B,T,C2)
        pos_embed = self.position_embedding_table(torch.arange(T))  # (T,C)
        x = tok_embed + pos_embed  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,C1)
        # B(atch) = batch_size = 4
        # T(ime) = block_size = 8
        # C(hannel)1 = vocab_size = 65
        # C(hannel)2 = n_embed = 32

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size):
        """
        TODO

        Parameters
        ----------
        idx : tensor
            is (B, T) array of indices in the current context

        max_new_tokes : int
            number of new tokens to be generated
        """
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
