# Week 4 Implementation — Causal Self-Attention

## 1. Token IDs and Data Preparation

A character-level tokenizer maps text to integer token IDs.

```python
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text, stoi):
    return [stoi[ch] for ch in text]

def decode(ids, itos):
    return "".join(itos[i] for i in ids)

def tokenize(text, stoi):
    return torch.tensor(encode(text, stoi), dtype=torch.long)
```

Token IDs use `torch.long` because they are used as indices into embedding tables.

### Train/validation split

Split the token stream before constructing overlapping context windows:

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

This avoids leakage from nearly identical overlapping windows.

---

## 2. Next-Token Batches

For context length `T = block_size`, each training example uses:

\[
X = (x_i,\ldots,x_{i+T-1})
\]

and target

\[
Y = (x_{i+1},\ldots,x_{i+T}).
\]

```python
def get_batch(data, batch_size, block_size):
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return x, y
```

### Dimensions

\[
X,Y \in \mathbb{N}^{B \times T}
\]

where:

- `B` = batch size
- `T` = sequence/context length

A useful correctness check is:

```python
assert torch.equal(x[:, 1:], y[:, :-1])
```

---

# 3. Single-Head Causal Self-Attention

Input:

\[
X \in \mathbb{R}^{B \times T \times d_{\text{model}}}
\]

For one head:

\[
Q=XW_Q,\qquad
K=XW_K,\qquad
V=XW_V.
\]

If the head dimension is \(d_h\),

\[
Q,K,V \in \mathbb{R}^{B \times T \times d_h}.
\]

Implementation:

```python
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size, block_size):
        super().__init__()

        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

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

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (k.shape[-1] ** 0.5)

        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        out = weights @ v

        return out
```

---

## 4. Attention Dimension Changes

Starting with:

\[
X:(B,T,d_{\text{model}})
\]

the linear projections give:

\[
Q,K,V:(B,T,d_h).
\]

Transpose the last two dimensions of \(K\):

```python
k.transpose(-2, -1)
```

so:

\[
K^\top:(B,d_h,T).
\]

Then:

\[
QK^\top:
(B,T,d_h)(B,d_h,T)
\rightarrow
(B,T,T).
\]

Thus the attention-score tensor has shape:

\[
S:(B,T,T).
\]

After softmax:

\[
A:(B,T,T).
\]

Finally:

\[
AV:
(B,T,T)(B,T,d_h)
\rightarrow
(B,T,d_h).
\]

So one attention head maps:

\[
\boxed{
(B,T,d_{\text{model}})
\rightarrow
(B,T,d_h)
}
\]

for every sequence in the batch independently.

There is no attention across batch elements.

---

# 5. Causal Mask

For a decoder-only Transformer, position \(i\) may attend only to positions

\[
j \le i.
\]

The mask:

```python
torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1,
)
```

marks positions above the diagonal as `True`.

Example:

\[
\begin{bmatrix}
0&1&1&1\\
0&0&1&1\\
0&0&0&1\\
0&0&0&0
\end{bmatrix}.
\]

Then:

```python
scores.masked_fill(mask, float("-inf"))
```

replaces forbidden future scores by \(-\infty\). After softmax, their probabilities become zero.

The diagonal is not masked, so a token may attend to itself.

### Why slice the stored mask?

The mask is created at maximum size `block_size`, but an input sequence may have length \(T < \text{block_size}\).

```python
mask = self.causal_mask[:T, :T]
```

reduces the mask to the same \(T\times T\) shape as the current attention scores.

---

# 6. Multi-Head Attention

With \(H\) heads,

\[
d_h = \frac{d_{\text{model}}}{H}.
\]

Each head receives the full input \(X\), but has its own learned \(W_Q,W_K,W_V\).

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()

        assert d_model % num_heads == 0

        head_size = d_model // num_heads

        self.heads = nn.ModuleList([
            AttentionHead(
                d_model=d_model,
                head_size=head_size,
                block_size=block_size,
            )
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        head_outputs = [
            head(x)
            for head in self.heads
        ]

        out = torch.cat(
            head_outputs,
            dim=-1,
        )

        return self.proj(out)
```

Each head produces:

\[
(B,T,d_h).
\]

Concatenating \(H\) heads along the last dimension gives:

\[
(B,T,Hd_h)
=
(B,T,d_{\text{model}}).
\]

The output projection \(W_O\) preserves this shape:

\[
\boxed{
\operatorname{MHA}:
(B,T,d_{\text{model}})
\rightarrow
(B,T,d_{\text{model}})
}
\]

This shape preservation is necessary for later residual connections.

---

# 7. Useful PyTorch Functions and Modules

## `nn.Module`

Base class for neural-network modules.

```python
class AttentionHead(nn.Module):
    ...
```

It automatically tracks submodules, parameters, buffers, device movement, and `state_dict()`.

---

## `nn.Linear(in_features, out_features)`

Applies a learned affine map to the last tensor dimension.

```python
self.query = nn.Linear(d_model, head_size, bias=False)
```

For:

\[
X:(B,T,d_{\text{model}})
\]

this produces:

\[
Q:(B,T,d_h).
\]

The batch and sequence dimensions are preserved.

---

## `nn.ModuleList`

Stores multiple PyTorch modules while registering their parameters correctly.

```python
self.heads = nn.ModuleList([
    AttentionHead(...)
    for _ in range(num_heads)
])
```

`ModuleList` does not define how the modules are connected. The forward logic is written manually.

It is useful when:

- the number of modules is configurable;
- modules are evaluated in parallel;
- custom iteration or branching is required.

For a simple sequential FNN, `nn.Sequential` is usually cleaner.

---

## `nn.Sequential`

Connects modules in a fixed chain.

```python
network = nn.Sequential(
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
)
```

Conceptually:

\[
X \rightarrow L_1 \rightarrow \text{ReLU} \rightarrow L_2.
\]

---

## `torch.cat`

Concatenates tensors along a selected dimension.

```python
out = torch.cat(head_outputs, dim=-1)
```

For four heads with:

\[
(B,T,8),
\]

concatenation gives:

\[
(B,T,32).
\]

`ModuleList` stores the heads; `torch.cat` combines their outputs.

---

## `torch.stack`

Creates a new dimension.

```python
x = torch.stack([...])
```

For batches, several vectors of shape `(T,)` become:

\[
(B,T).
\]

Difference:

- `torch.cat` joins along an existing dimension.
- `torch.stack` introduces a new dimension.

---

## `torch.transpose`

```python
k.transpose(-2, -1)
```

switches the final two dimensions:

\[
(B,T,d_h)
\rightarrow
(B,d_h,T).
\]

---

## `torch.triu`

Returns the upper triangular part of a tensor.

```python
torch.triu(mask, diagonal=1)
```

is convenient for constructing a causal future-token mask.

---

## `masked_fill`

```python
scores.masked_fill(mask, float("-inf"))
```

replaces elements wherever the boolean mask is `True`.

---

## `F.softmax`

```python
weights = F.softmax(scores, dim=-1)
```

normalizes each attention row:

\[
\sum_j A_{bij}=1.
\]

---

## `register_buffer`

Stores a tensor as part of a module without making it trainable.

```python
self.register_buffer("causal_mask", mask)
```

The mask:

- moves with the model to CPU/GPU;
- appears in `state_dict()`;
- does not appear in `model.parameters()`.

---

## `torch.ones_like`

There is no `nn.one_like`. The PyTorch function is:

```python
torch.ones_like(x)
```

It creates a tensor of ones with the same shape, dtype, and device as `x`.

Example:

```python
row_sums = weights.sum(dim=-1)

assert torch.allclose(
    row_sums,
    torch.ones_like(row_sums),
)
```

If:

\[
\text{row\_sums}:(B,T),
\]

then:

\[
\texttt{torch.ones\_like(row\_sums)}:(B,T).
\]

---

## `torch.allclose`

Checks approximate numerical equality.

```python
torch.allclose(a, b, atol=1e-6)
```

This is preferable to exact equality for floating-point tensors.

---

# 8. Attention Tests

Important tests include:

### Output shape

```python
assert out.shape == (B, T, head_size)
```

For multi-head attention:

```python
assert out.shape == x.shape
```

### Attention rows sum to one

```python
row_sums = weights.sum(dim=-1)

assert torch.allclose(
    row_sums,
    torch.ones_like(row_sums),
    atol=1e-6,
)
```

### Future-token weights are zero

```python
future_mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1,
)

assert torch.all(
    weights[0][future_mask] == 0
)
```

### Functional causality

Changing a future token must not change earlier outputs.

```python
x2 = x.clone()
x2[:, -1, :] += 1000

out1 = head(x)
out2 = head(x2)

assert torch.allclose(
    out1[:, :-1, :],
    out2[:, :-1, :],
    atol=1e-5,
)
```

### Gradient flow

```python
out = mha(x)
loss = out.sum()
loss.backward()

for parameter in mha.parameters():
    assert parameter.grad is not None
```

---

# 9. Week 4 Pipeline

The implementation now covers:

\[
\text{text}
\rightarrow
\text{token IDs}
\rightarrow
(X,Y)
\rightarrow
X:(B,T,d_{\text{model}})
\]

followed by:

\[
X
\rightarrow
Q,K,V
\rightarrow
\frac{QK^\top}{\sqrt{d_h}}
\rightarrow
\text{causal mask}
\rightarrow
A
\rightarrow
AV
\]

and, for multiple heads:

\[
H^{(1)},\ldots,H^{(H)}
\rightarrow
\text{concatenate}
\rightarrow
W_O.
\]

Week 5 will add the remaining Transformer-block components: LayerNorm, residual connections, FFN, and block stacking.
