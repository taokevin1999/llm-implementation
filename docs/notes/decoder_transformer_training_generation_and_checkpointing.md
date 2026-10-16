# Decoder Transformer: Training, Generation, and Checkpointing

A sanitized implementation reference covering a complete decoder-only Transformer, evaluation, autoregressive generation, and training recovery. No personal usernames, repository URLs, or machine-specific home paths are included.

## 1. Project organization and notation

Run commands from the repository root with the project environment activated. The Python package is `llm_impl`; import from `llm_impl`, not `src.llm_impl`.

| File | Responsibility |
|---|---|
| `src/llm_impl/data.py` | Existing character tokenizer and next-token batching |
| `src/llm_impl/attention.py` | Existing causal multi-head attention |
| `src/llm_impl/embeddings.py` | Token and learned position embeddings |
| `src/llm_impl/blocks.py` | FFN and Pre-LayerNorm Transformer block |
| `src/llm_impl/model.py` | Stacked decoder, output logits, cross-entropy |
| `src/llm_impl/generation.py` | Autoregressive generation |
| `src/llm_impl/checkpoint.py` | Checkpoint saving |
| `scripts/check_transformer_block.py` | Block shape, evaluation, and causality checks |
| `scripts/check_decoder.py` | Decoder shape, gradient, and uniform-loss checks |
| `scripts/overfit_batch.py` | Fixed-batch memorization check |
| `scripts/train_decoder.py` | Text training, evaluation, logging, and resume |
| `scripts/generate_text.py` | Checkpoint loading and saved generation samples |
| `tests/test_embeddings.py` | Embedding behavior and context-limit checks |
| `tests/test_checkpoint.py` | Checkpoint prediction recovery |
| `docs/notes/` | Course references |

`storage/` is the established symlink to the external SSD. Data, checkpoints, and results go through it; source code and the environment stay in the Linux filesystem. Existing run paths retain `week5` for compatibility with the scripts.

| Symbol | Meaning |
|---|---|
| B | Batch size |
| T | Current input sequence length |
| C | Embedding dimension (`n_embed`, or `d_model` in attention) |
| V | Vocabulary size |
| L | Number of Transformer blocks |
| T_max | Maximum context length (`block_size`) |

The existing attention constructor is:

```python
MultiHeadAttention(d_model, num_heads, block_size)
```

Require `d_model % num_heads == 0`. Its input and output are both `(B, T, C)`.

## 2. Embeddings: token IDs to representations

Learn two tables: E of shape (V, C), and P of shape (T_max, C).

For token ID x[b,t], the initial representation is:

`h[b,t,:] = E[x[b,t],:] + P[t,:]`.

Create `src/llm_impl/embeddings.py`:

```python
import torch
from torch import nn


class TransformerInput(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)

    def forward(self, idx):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("Sequence length exceeds block_size")

        positions = torch.arange(T, device=idx.device)
        tokens = self.token_embedding(idx)          # (B, T, C)
        positions = self.position_embedding(positions)  # (T, C)
        return tokens + positions                  # (B, T, C)
```

`nn.Embedding` creates a trainable lookup table. Calling it selects rows using integer IDs; no explicit one-hot encoding is needed. Position vectors broadcast across B.

Same token + same position gives the same initial vector across sequences. Same token + different positions generally gives different vectors. These representations are not yet contextualized.

Useful checks:

- Output shape is (B, T, C).
- For repeated token a at positions i and j, h[i] − h[j] = P[i] − P[j].
- T greater than block_size raises ValueError.

## 3. FFN, LayerNorm, and residual block

Use Pre-LayerNorm:

`U = X + MHA(LN1(X))`

`Y = U + FFN(LN2(U))`

Create `src/llm_impl/blocks.py`:

```python
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
```

| Operation | Shape |
|---|---|
| LayerNorm / attention / residual addition | (B, T, C) → (B, T, C) |
| FFN first linear | (B, T, C) → (B, T, 4C) |
| GELU | (B, T, 4C) → (B, T, 4C) |
| FFN second linear | (B, T, 4C) → (B, T, C) |

Attention mixes information between tokens. The FFN mixes features within each token, sharing its parameters over all tokens and sequences.

### Module construction versus computation

```python
ln = nn.LayerNorm(C)  # Construct a module and its parameters
normalized = ln(x)   # Compute its forward operation
```

For each (b,t), LayerNorm automatically computes the mean and variance over C, centers/scales the vector with epsilon for stability, and applies trainable coordinatewise gamma and beta. Gamma starts at ones and beta at zeros. Input statistics are computed in both train and evaluation modes; LN1 and LN2 have separate learned parameters.

`attention` is a constructor argument containing a module instance:

```python
block = TransformerBlock(
    attention=MultiHeadAttention(d_model=128, num_heads=4, block_size=64),
    n_embed=128,
    dropout=0.1,
)
```

Pass `MultiHeadAttention(...)`, not the class name alone. Assigning it to `self.attention` registers the same object as a child module. Its parameters are included in the block's `parameters()` and `state_dict()`. Calling the block later supplies the tensor x.

The supplied attention module should include its output projection, but not another residual wrapper.

`nn.Sequential` registers modules and executes them in order. `nn.ModuleList` registers modules but requires an explicit forward loop.

### Block checks

In evaluation mode, check output shape, finite values, repeated-call agreement, and causality. Change only the final input token: earlier outputs should remain unchanged. LayerNorm and the FFN act tokenwise and should preserve causal behavior.

## 4. Complete decoder and loss

Create `src/llm_impl/model.py`:

```python
from torch import nn
from torch.nn import functional as F
from llm_impl.embeddings import TransformerInput
from llm_impl.blocks import TransformerBlock


class DecoderLM(nn.Module):
    def __init__(
        self, vocab_size, n_embed, block_size, attentions, dropout=0.0
    ):
        super().__init__()
        self.block_size = block_size
        self.embedding = TransformerInput(vocab_size, n_embed, block_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(attention, n_embed, dropout)
            for attention in attentions
        ])
        self.final_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
            )
        return logits, loss
```

Pipeline: IDs (B,T) → embeddings (B,T,C) → L blocks (B,T,C) → final normalization (B,T,C) → vocabulary logits (B,T,V).

Construct independent attention modules:

```python
attentions = [
    MultiHeadAttention(d_model=C, num_heads=4, block_size=block_size)
    for _ in range(n_layers)
]
```

Do not use `[attention] * n_layers`: that repeats references to one object and shares attention parameters.

### Vocabulary and target alignment

For the synthetic check, explicitly define `V = 10`; valid IDs are 0 through 9. For text, use `V = len(stoi)`.

For tokens [a,b,c,d,e], input is [a,b,c,d] and targets are [b,c,d,e]. The batching function already shifts targets; the decoder must not shift them again.

Cross-entropy is the mean negative log-probability of the correct next token over BT predictions. Pass raw logits to `F.cross_entropy`; it handles log-softmax internally.

`reshape(B*T, V)` merges the batch and sequence axes while keeping each token's V scores together. Reshape targets to (B*T,) in the same order.

For uniform logits, every token has probability 1/V:

- loss = log(V), with natural logarithm;
- perplexity = exp(loss) = V.

```python
uniform_logits = torch.zeros_like(logits)
uniform_loss = F.cross_entropy(
    uniform_logits.reshape(-1, V),
    targets.reshape(-1),
)
torch.testing.assert_close(
    uniform_loss,
    uniform_loss.new_tensor(math.log(V)),
)
```

`uniform_loss.new_tensor(value)` creates a new tensor with that value and the same dtype/device as uniform_loss. It neither modifies uniform_loss nor copies its value. A randomly initialized model need not have exactly uniform logits.

Also verify finite scalar loss, logits shape, and nonzero finite gradients reaching the token embedding after `loss.backward()`.

## 5. Training and evaluation

### Fixed-batch check

Before full training, train a fresh model repeatedly on one sequence, with dropout and weight decay set to zero:

```python
tokens = torch.tensor([[2, 5, 2, 7, 1]], device=device)
idx, targets = tokens[:, :-1], tokens[:, 1:]

optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=0.0
)

model.train()
for step in range(300):
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(idx, targets)
    loss.backward()
    optimizer.step()
```

Move the model to its device before constructing the optimizer. Keep model and input tensors on the same device. Create the optimizer once, outside the loop.

Loss should fall substantially; the target predictions are [[5,2,7,1]]. Memorization verifies the learning pipeline, not generalization. Start the real text run from fresh parameters.

### Data and initial configuration

Dataset: [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).

Store it at `storage/datasets/tinyshakespeare/input.txt`. Split the text into the first 90% for training and last 10% for validation before sampling windows.

Use existing `build_vocab(train_text)`, `tokenize(text, stoi)`, and `get_batch(data, batch_size, block_size)`. Build vocabulary on training text; explicitly reject unseen validation characters because this tokenizer has no unknown-token mechanism. Use the same mapping for both splits.

Initial settings:

| Setting | Value |
|---|---|
| Embedding dimension / heads / blocks | 128 / 4 / 3 |
| Context length / batch size | 64 / 16 |
| FFN dropout | 0.1 |
| AdamW learning rate / weight decay | 3e-4 / 0.01 |
| Training updates / evaluation interval | 1000 / 100 |
| Evaluation batches per split | 20 |
| Main seed / train-eval seed / val-eval seed | 42 / 123 / 456 |

The existing attention module has no dropout argument. The configured dropout applies in the FFNs.

Sample fresh training windows each update. Prepare fixed evaluation windows once, using separate `torch.Generator().manual_seed(seed)` instances. This makes checkpoint comparisons consistent without advancing the training sampler's random sequence.

### Token-weighted evaluation

```python
def evaluate(model, batches, device):
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0

    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            n = y.numel()
            total_nll += loss.item() * n
            total_tokens += n

    model.train(was_training)
    mean_loss = total_nll / total_tokens
    return mean_loss, math.exp(mean_loss), total_tokens
```

Aggregate loss as sum(N_k * loss_k) / sum(N_k), then exponentiate. Do not average batch perplexities. Our batches contain no padding or ignored targets; if those are introduced, count only scored tokens.

The evaluation token count includes repeated occurrences when sampled windows overlap. This is an estimate on fixed windows, not exhaustive validation-set scoring.

| Operation | Meaning |
|---|---|
| `model.train()` | Enable training behavior, including dropout |
| `model.eval()` | Disable dropout; does not disable autograd |
| `with torch.no_grad():` | Disable gradient recording in the enclosed computation |
| `zero_grad(set_to_none=True)` | Clear old gradients |
| `loss.backward()` | Compute current gradients |
| `optimizer.step()` | Update parameters |

Use evaluation mode for both reported training and validation losses. The optimizer's per-batch loss is computed before its corresponding update.

### Run folders and logging

Each invocation, including resume, creates a new timestamp:

```python
run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
run_dir = Path("storage/outputs/week5") / run_name
run_dir.mkdir(parents=True, exist_ok=False)
```

`Path / name` joins paths; `parents=True` creates missing parents; `exist_ok=False` rejects an existing run folder.

Save `metadata.json`: configuration, tokenizer mapping, dataset path and SHA-256 fingerprint, split index, token counts, seeds, software version, start/target steps, and source checkpoint. A fingerprint identifies data contents; it is not a data backup.

`json.dumps(metadata, indent=2)` serializes a dictionary into readable JSON. CSV rows record step, train loss, validation loss, validation perplexity, and scored validation tokens. `file.flush()` flushes Python's output buffer to the OS, not necessarily to physical disk immediately.

## 6. Generation

Create `src/llm_impl/generation.py`:

```python
import torch


def generate(model, idx, max_new_tokens, temperature=1.0):
    if idx.ndim != 2 or idx.shape[1] == 0:
        raise ValueError("idx must have shape (B, T) with T >= 1")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = idx[:, -model.block_size:]
            logits, _ = model(context)
            next_logits = logits[:, -1, :]  # (B, V)

            if temperature == 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(
                    next_logits / temperature, dim=-1
                )
                next_id = torch.multinomial(probabilities, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)

    model.train(was_training)
    return idx
```

The final-position logits predict the next token. Sampling returns (B,1); concatenating grows (B,T) to (B,T+1). Return value includes the prompt.

For positive temperature tau, probabilities are softmax(logits/tau). Lower tau concentrates probabilities; higher tau flattens them. Temperature zero is handled explicitly as greedy decoding.

`torch.multinomial(probabilities, 1)` makes one categorical draw per row. Evaluation mode disables dropout but does not disable this intentional sampling randomness.

### Context cropping and slicing

`idx[:, -block_size:]` returns the latest min(T, block_size) tokens. Out-of-range slice bounds clip automatically:

```python
x = [10, 20, 30]
x[-64:]  # All three entries
# x[-64] raises IndexError: single-index access differs from slicing.
```

This follows Python slicing semantics. PyTorch supports the multidimensional comma-separated form; ordinary lists do not.

The generated sequence can exceed block_size, but each forward pass sees only the cropped window. Learned positions restart from zero within that window. We recompute the window without a KV cache.

### Load, generate, and save samples

Load the dictionary with `torch.load(path, map_location="cpu", weights_only=True)`. Reconstruct the architecture from its config, call `load_state_dict(saved["model_state"])`, move to the device, then use `eval()`.

`weights_only=True` permits the tensor/primitive dictionary checkpoint; it does not select only the dictionary's model_state entry. Python class definitions are still required.

Reconstruct `itos = {i: ch for ch, i in stoi.items()}`. Encode the prompt into (1,T), generate, and decode the first output row.

Fixed comparison settings: prompts `"ROMEO:\n"` and `"JULIET:\n"`; temperatures 0, 0.7, 1.0; 300 new tokens; seed 42 reset immediately before each generation. Save checkpoint path/step, prompt, temperature, seed, length, environment, and generated text in a timestamped JSON under the run's output folder.

Our tokenizer is character-level and has no EOS token. The loop stops at max_new_tokens. Fixed seeds aid repeatability in the same environment, not guaranteed equality across hardware/software.

## 7. Checkpoint saving

Create `src/llm_impl/checkpoint.py`:

```python
from pathlib import Path
import torch


def save_checkpoint(
    path, model, optimizer, config, stoi, step, metadata
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "stoi": stoi,
        "step": step,
        "metadata": metadata,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        ),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
```

Save at evaluation points before sampling the next batch:

```python
save_checkpoint(
    checkpoint_dir / f"step_{step:06d}.pt",
    model, optimizer, config, stoi, step, metadata,
)
```

Use `storage/checkpoints/week5/<RUN_NAME>/`. Save `last.pt` after the loop. `:06d` gives six-digit, zero-padded step numbers.

Writing a temporary file before replacement reduces exposure to partially written destination files. No claim of crash-proof durability is implied.

| Saved entry | Role |
|---|---|
| model_state | Parameters and registered buffers |
| optimizer_state | AdamW moments, counters, and group settings |
| config / stoi | Architecture/settings and exact token mapping |
| step | Completed optimizer updates |
| metadata | Data identity, split, and experiment provenance |
| CPU / CUDA RNG states | Current random sequence positions |

A seed starts a random sequence; an RNG state records where it currently is. Current training uses CPU randomness for batching and GPU randomness for CUDA dropout. Python/NumPy RNG states would also be needed if those libraries were used for stochastic training operations.

Gradients are unnecessary here: checkpoints are between updates, and the next iteration clears gradients.

## 8. Resume training: final control flow

These are the consolidated additions to `scripts/train_decoder.py`, not a separate training program.

### Parse arguments and load before model construction

```python
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=Path, default=None)
parser.add_argument("--steps", type=int, default=None)
args = parser.parse_args()

saved = None
if args.resume is not None:
    saved = torch.load(
        args.resume, map_location="cpu", weights_only=True
    )
    required = {"cpu_rng_state", "cuda_rng_states", "metadata"}
    missing = required - saved.keys()
    if missing:
        raise ValueError(f"Missing recovery information: {sorted(missing)}")
```

After loading/tokenizing the data and defining the default config, but before constructing the model:

```python
if saved is not None:
    config = saved["config"].copy()
    if stoi != saved["stoi"]:
        raise ValueError("Tokenizer mapping differs")
    if V != config["vocab_size"]:
        raise ValueError("Vocabulary size differs")

start_step = 0 if saved is None else saved["step"]
target_steps = config["steps"] if args.steps is None else args.steps
if target_steps <= start_step:
    raise ValueError("Target must exceed completed updates")
config["steps"] = target_steps
```

Construct the model with this config, move it to device, and construct the optimizer. Then load:

```python
if saved is not None:
    model.load_state_dict(saved["model_state"])
    optimizer.load_state_dict(saved["optimizer_state"])
```

Optimizer loading also restores saved learning rates and other group settings.

Build fixed evaluation batches and metadata. Compare the current dataset SHA-256 and split index with saved metadata; reject mismatches. Record `start_step`, `target_steps`, and `resumed_from` in the new run's metadata.

### Restore RNG states last

After setup, immediately before training:

```python
if saved is not None:
    cuda_states = saved["cuda_rng_states"]
    if device.type == "cuda":
        if cuda_states is None:
            raise ValueError("Missing CUDA RNG state")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs")
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states is not None:
        raise ValueError("Resume on the original CUDA setup")
    torch.set_rng_state(saved["cpu_rng_state"])

model.train()
```

Model initialization consumes randomness, so restore RNG state after construction. Do not reseed afterward. These checks deliberately keep this exercise on the original CPU/CUDA setup.

### Loop semantics

```python
for step in range(start_step, target_steps + 1):
    if (
        step == start_step
        or step % config["eval_interval"] == 0
        or step == target_steps
    ):
        # Evaluate fixed train/validation batches.
        # Write/flush a metrics row; print it.
        # Save step_{step:06d}.pt.
        pass

    if step == target_steps:
        break

    x, y = get_batch(
        train_data, config["batch_size"], config["block_size"]
    )
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss at step {step}")
    loss.backward()
    optimizer.step()
```

The commented evaluation branch stands for the existing evaluation/logging/save code; retain that code rather than replacing it with pass.

Step means completed updates. At step 500, evaluate/save the state after 500 updates, then perform update 501. At the target, evaluate/save and break without another update. Save final last.pt with `step=target_steps`.

Repeated starting evaluation does not consume the training RNG in our implementation: batches are already prepared and dropout is disabled.

### Commands and clarifications

```bash
python scripts/train_decoder.py

python scripts/train_decoder.py \
  --resume storage/checkpoints/week5/<RUN_NAME>/step_000500.pt \
  --steps 2000
```

Replace `<RUN_NAME>` with the actual folder name. The second command performs 1,500 additional updates, reaching 2,000 total.

- `args.steps` is created by `add_argument("--steps", ...)` and `parse_args()`.
- The leading `--` is part of command-line option syntax, but is removed from the default attribute name.
- Internal hyphens become underscores: `--max-new-tokens` → `args.max_new_tokens`.
- Omitting --steps uses the checkpoint's saved target. Resuming a completed last.pt therefore requires a larger target.
- A new timestamped metrics/checkpoint folder is created on every execution, including resume. Original files remain intact; resumed_from records lineage.
- Old checkpoints without RNG states still support generation and ordinary state loading, but this reproducible-resume path rejects them.
- RNG restoration is necessary but insufficient for guaranteed bitwise replay: use the same code, data, environment, and deterministic operations.

## 9. Recovery test belongs under tests/

Remove the inline deepcopy/recovery check from the end of `train_decoder.py`; retain final checkpoint saving and path printing.

Create `tests/test_checkpoint.py`:

```python
import copy
import torch

from llm_impl.attention import MultiHeadAttention
from llm_impl.checkpoint import save_checkpoint
from llm_impl.model import DecoderLM


def test_checkpoint_restores_predictions(tmp_path):
    torch.manual_seed(42)
    config = {
        "vocab_size": 10, "n_embed": 16, "num_heads": 2,
        "n_layers": 2, "block_size": 8, "dropout": 0.1,
    }
    attentions = [
        MultiHeadAttention(16, 2, 8) for _ in range(2)
    ]
    model = DecoderLM(10, 16, 8, attentions, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters())
    probe = torch.tensor([[2, 5, 2, 7], [1, 3, 4, 6]])
    model.eval()

    with torch.no_grad():
        expected, _ = model(probe)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model, optimizer, config,
        {str(i): i for i in range(10)}, 0,
        {"purpose": "checkpoint test"},
    )
    saved = torch.load(path, map_location="cpu", weights_only=True)
    restored = copy.deepcopy(model)

    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
        changed, _ = restored(probe)
    assert not torch.allclose(changed, expected)

    restored.load_state_dict(saved["model_state"])
    restored.eval()
    with torch.no_grad():
        actual, _ = restored(probe)
    torch.testing.assert_close(actual, expected)
```

`tmp_path` is a temporary directory supplied automatically by pytest. The model and inputs stay on CPU; this test requires neither trained weights nor an SSD. Random initialized parameters are sufficient to test a save/load round trip.

```bash
python -m pytest tests/test_checkpoint.py -q
```

Prediction recovery and training continuation are different checks. For continuation, resume an intermediate checkpoint to the original final step and compare fixed-evaluation metrics along the trajectory. Preserve the original run for comparison.

## 10. Final comparison and scope

Use the same prompts, temperatures, seeds, tokenizer, and validation windows for early/late checkpoint comparisons. Record:

| Checkpoint | Completed updates | Validation loss | Validation perplexity | Sample observations |
|---|---:|---:|---:|---|
| Early | Fill from run | Measured | Measured | Spelling, repetition, structure |
| Late | Fill from run | Measured | Measured | Same criteria |

Do not infer model quality from one attractive sample. No training results are claimed in this reference.

This implementation uses learned absolute positions, LayerNorm, GELU FFNs, character tokenization, and uncached generation. RoPE, RMSNorm, SwiGLU, more extensive evaluation/profiling, and later PEFT/RAG/serving topics remain subsequent material. No bigram model is required.

## References

- [PyTorch Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)
- [PyTorch LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)
- [PyTorch cross-entropy](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html)
- [PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
- [PyTorch multinomial sampling](https://docs.pytorch.org/docs/2.14/generated/torch.multinomial.html)
- [PyTorch checkpoint tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

