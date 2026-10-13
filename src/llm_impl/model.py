from torch import nn
from torch.nn import functional as F

from llm_impl.embeddings import TransformerInput
from llm_impl.blocks import TransformerBlock


class DecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embed,
        block_size,
        attentions,
        dropout=0.0,
    ):
        super().__init__()

        self.block_size = block_size

        self.embedding = TransformerInput(
            vocab_size=vocab_size,
            n_embed=n_embed,
            block_size=block_size,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                attention=attention,
                n_embed=n_embed,
                dropout=dropout,
            )
            for attention in attentions
        ])

        self.final_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)       # (B, T, C)

        for block in self.blocks:
            x = block(x)              # (B, T, C)

        x = self.final_norm(x)         # (B, T, C)
        logits = self.lm_head(x)       # (B, T, V)

        loss = None

        if targets is not None:
            B, T, V = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
            )

        return logits, loss