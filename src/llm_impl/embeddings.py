import torch
from torch import nn


class TransformerInput(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size):
        super().__init__()

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)

    def forward(self, idx):
        # idx: (B, T) shaped tensor of integer token IDs
        # C = embedding dim = n_embed
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError("Sequence length exceeds block_size")

        positions = torch.arange(T, device=idx.device)

        token_vectors = self.token_embedding(idx)              # (B, T, C)
        position_vectors = self.position_embedding(positions)  # (T, C)

        return token_vectors + position_vectors                # (B, T, C)