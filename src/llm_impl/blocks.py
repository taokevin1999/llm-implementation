from torch import nn


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, attention, n_embed, dropout=0.0):
        super().__init__()

        self.attention = attention
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x