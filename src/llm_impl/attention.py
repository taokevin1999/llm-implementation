import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size, block_size):
        super().__init__()

        self.key = nn.Linear(
            d_model,
            head_size,
            bias=False,
        )

        self.query = nn.Linear(
            d_model,
            head_size,
            bias=False,
        )

        self.value = nn.Linear(
            d_model,
            head_size,
            bias=False,
        )

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    block_size,
                    block_size,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )

    def forward(self, x, return_weights=False):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)

        scores = scores / (k.shape[-1] ** 0.5)

        mask = self.causal_mask[:T, :T]

        scores = scores.masked_fill(
            mask,
            float("-inf"),
        )

        weights = F.softmax(
            scores,
            dim=-1,
        )

        out = weights @ v
        if return_weights:
            return out,weights

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,block_size):
        super().__init__()

        assert d_model % num_heads == 0

        head_size = d_model // num_heads

        self.heads = nn.ModuleList([AttentionHead(d_model=d_model,head_size=head_size,block_size=block_size,)for _ in range(num_heads)])

        self.proj = nn.Linear(
            d_model,
            d_model,
        )

    def forward(self, x):
        head_outputs = [
            head(x)
            for head in self.heads
        ]

        out = torch.cat(
            head_outputs,
            dim=-1,
        )

        out = self.proj(out)

        return out

head = AttentionHead(
    d_model=8,
    head_size=4,
    block_size=5,
)
for name, param in head.named_parameters():
    print(name)
