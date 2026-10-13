from llm_impl.attention import MultiHeadAttention
from llm_impl.blocks import TransformerBlock
import torch

B,T=2,4
C=6

#must have T<=blocksize
mha = MultiHeadAttention(
        d_model=C,
        num_heads=2,
        block_size=T,
    )

block = TransformerBlock(
    attention=mha,
    n_embed=C,
    dropout=0.1,
)

x = torch.randn(B, T, C)

block.eval()

with torch.no_grad():
    y1 = block(x)
    y2 = block(x)

assert y1.shape == x.shape
assert torch.isfinite(y1).all()
torch.testing.assert_close(y1, y2)

# Change only the final token.
x_changed = x.clone()
x_changed[:, -1, :] += torch.randn_like(x_changed[:, -1, :])

with torch.no_grad():
    y_changed = block(x_changed)

# Earlier outputs must not depend on a future token.
torch.testing.assert_close(
    y1[:, :-1, :],
    y_changed[:, :-1, :],
)

print("Shape, evaluation repeatability, and causality checks passed.")