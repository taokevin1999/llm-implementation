import math
import torch
from torch.nn import functional as F

from llm_impl.attention import MultiHeadAttention
from llm_impl.model import DecoderLM

n_layers = 3
C=16
num_head=4
block_size=10
T=7
B=3
V=10

attentions = [
    MultiHeadAttention(C,num_head,block_size) 
    for _ in range(n_layers)
]

model = DecoderLM(
    vocab_size=V,
    n_embed=C,
    block_size=block_size,
    attentions=attentions,
    dropout=0.1,
)

# Synthetic IDs for checking the implementation.
tokens = torch.randint(V, (B, T + 1))
idx = tokens[:, :-1]
targets = tokens[:, 1:]

logits, loss = model(idx, targets)

assert logits.shape == (B, T, V)
assert loss.ndim == 0
assert torch.isfinite(loss)

model.zero_grad(set_to_none=True)
loss.backward()

# Confirm the loss reaches the input embedding parameters.
grad = model.embedding.token_embedding.weight.grad
assert grad is not None
assert torch.isfinite(grad).all()
assert grad.abs().sum() > 0

print("Logits shape:", logits.shape)
print("Loss:", loss.item())
print("Gradient flow check passed.")

# verifying uniform pred baseline
uniform_logits = torch.zeros_like(logits)

uniform_loss = F.cross_entropy(
    uniform_logits.reshape(B * T, V),
    targets.reshape(B * T),
)

torch.testing.assert_close(
    uniform_loss,
    uniform_loss.new_tensor(math.log(V)),
)

print("Uniform loss:", uniform_loss.item())
print("Expected log(V):", math.log(V))
print("Uniform perplexity:", uniform_loss.exp().item())