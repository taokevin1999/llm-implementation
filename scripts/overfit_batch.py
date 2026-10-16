import math
import torch
from torch.nn import functional as F

from llm_impl.attention import MultiHeadAttention
from llm_impl.model import DecoderLM

# experiment parameters
block_size=6
n_layers=3
V=10
C=8
num_head=2

torch.manual_seed(42)

# setup model
attentions = [
    MultiHeadAttention(C,num_head,block_size) 
    for _ in range(n_layers)
]

model = DecoderLM(
    vocab_size=V,
    n_embed=C,
    block_size=block_size,
    attentions=attentions,
    dropout=0.0,
)

# move to GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

# One fixed sequence, with next-token targets.
tokens = torch.tensor(
    [[2, 5, 2, 7, 1]],
    dtype=torch.long,
    device=device,
)

idx = tokens[:, :-1]      # [[2, 5, 2, 7]]
targets = tokens[:, 1:]   # [[5, 2, 7, 1]]

print("Device:", device)

# create optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.0,
)

#set to training mode
model.train()

with torch.no_grad():
    _, initial_loss = model(idx, targets)

print(f"Initial loss: {initial_loss.item():.4f}")

#actual training steps
for step in range(1, 301):
    optimizer.zero_grad(set_to_none=True)

    logits, loss = model(idx, targets)

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss at step {step}")

    loss.backward()
    optimizer.step()

    if step == 1 or step % 50 == 0:
        print(f"Step {step:3d} | loss {loss.item():.4f}")

# evaluate after training
model.eval()

with torch.no_grad():
    logits, final_loss = model(idx, targets)
    predictions = logits.argmax(dim=-1)

    accuracy = (
        predictions == targets
    ).float().mean().item()

print("Targets:    ", targets.cpu().tolist())
print("Predictions:", predictions.cpu().tolist())
print(f"Final loss: {final_loss.item():.4f}")
print(f"Perplexity: {final_loss.exp().item():.4f}")
print(f"Accuracy:   {accuracy:.1%}")