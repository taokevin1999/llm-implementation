# Tensor Operations Reference

This note consolidates the tensor operations used in the implementation course. It is intended as a compact reference rather than a lesson transcript.

## 1. Core Transformer Tensor Setup

Use one Transformer-shaped example throughout:

```python
import torch

torch.manual_seed(42)

B = 2
T = 5
d_model = 12
H = 3
d_head = d_model // H

assert d_model % H == 0

x = torch.randn(B, T, d_model)

W_Q = torch.randn(d_model, d_model, requires_grad=True)
W_K = torch.randn(d_model, d_model, requires_grad=True)
W_V = torch.randn(d_model, d_model, requires_grad=True)

Q = x @ W_Q
K = x @ W_K
V = x @ W_V
```

Shapes:

```text
x              (B, T, d_model) = (2, 5, 12)
W_Q, W_K, W_V  (d_model, d_model) = (12, 12)
Q, K, V        (B, T, d_model) = (2, 5, 12)
```

For `x[b, t, j]`:

- `b`: batch index
- `t`: token position
- `j`: feature coordinate

---

## 2. Indexing and Slicing

```python
x[0]          # (T, d_model)
x[:, 2]       # (B, d_model)
x[:, 2:3]     # (B, 1, d_model)
x[:, :, 0]    # (B, T)
x[:, :3, :]   # (B, 3, d_model)
```

Important distinction:

```python
x[:, 2, :]        # (B, d_model)
x[:, 2:3, :]      # (B, 1, d_model)
```

Integer indexing removes the selected axis; slicing preserves it.

---

## 3. `unsqueeze`, `squeeze`, and `None`

Suppose one scalar is associated with each batch item:

```python
scale = torch.randn(B)          # (B,)
scale = scale.unsqueeze(1)      # (B, 1)
scale = scale.unsqueeze(2)      # (B, 1, 1)

scaled_x = x * scale            # broadcasts to (B, T, d_model)
```

Equivalent shorthand:

```python
scale = torch.randn(B)
scaled_x = x * scale[:, None, None]
```

Removing a size-1 dimension:

```python
y = x[:, 2:3, :]      # (B, 1, d_model)
y = y.squeeze(1)      # (B, d_model)
```

Prefer specifying the axis explicitly rather than using unrestricted `squeeze()`.

---

## 4. `reshape`, `view`, `transpose`, and `permute`

Split the model dimension into attention heads:

```python
Q = Q.reshape(B, T, H, d_head)
K = K.reshape(B, T, H, d_head)
V = V.reshape(B, T, H, d_head)
```

Shape:

```text
(B, T, d_model)
-> (B, T, H, d_head)
```

Move the head dimension before the token dimension:

```python
Q = Q.transpose(1, 2)
K = K.transpose(1, 2)
V = V.transpose(1, 2)
```

Shape:

```text
(B, T, H, d_head)
-> (B, H, T, d_head)
```

Equivalent with `permute`:

```python
Q = Q.permute(0, 2, 1, 3)
```

Use:

- `transpose(i, j)` to swap two axes
- `permute(...)` to specify an arbitrary axis ordering
- `reshape(...)` as the default shape-changing operation
- `view(...)` only when the memory layout is compatible

After `transpose` or `permute`, a tensor may be non-contiguous:

```python
Q.is_contiguous()
```

If code specifically requires a contiguous layout:

```python
Q = Q.contiguous()
```

Typical pattern seen in existing PyTorch code:

```python
out = out.transpose(1, 2).contiguous().view(B, T, d_model)
```

For this course, prefer:

```python
out = out.transpose(1, 2).reshape(B, T, d_model)
```

---

## 5. Broadcasting

PyTorch compares shapes from right to left. Dimensions are compatible when they are equal or one of them is `1`.

### Feature-wise parameters

LayerNorm-style parameters:

```python
gamma = torch.ones(d_model)     # (d_model,)
beta = torch.zeros(d_model)     # (d_model,)

y = gamma * x + beta            # (B, T, d_model)
```

Conceptually:

```text
(d_model)
-> (1, 1, d_model)
-> (B, T, d_model)
```

### Batch-wise scaling

```python
scale = torch.randn(B, 1, 1)
y = x * scale
```

Shapes:

```text
(B, T, d_model)
(B, 1, 1)
-> (B, T, d_model)
```

### Attention mask

After attention scores are computed:

```python
scores = Q @ K.transpose(-2, -1)
```

Shape:

```text
(B, H, T, T)
```

Create one causal mask:

```python
mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1
)
```

Here `True` means blocked.

```python
scores = scores.masked_fill(mask, float("-inf"))
```

The `(T, T)` mask broadcasts across batch and head dimensions.

For Boolean masks:

```python
~mask
```

means logical NOT.

`masked_fill(mask, value)` modifies entries where `mask == True`.

---

## 6. Reductions and `dim=`

For:

```python
x.shape == (B, T, d_model)
```

examples:

```python
x.mean(dim=0)       # (T, d_model), reduce batch
x.mean(dim=1)       # (B, d_model), reduce tokens
x.mean(dim=-1)      # (B, T), reduce features
```

`dim=k` means: combine values along axis `k`.

### `keepdim=True`

Useful when the reduced result must broadcast back against the original tensor:

```python
mu = x.mean(dim=-1, keepdim=True)
var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)

x_hat = (x - mu) / torch.sqrt(var + 1e-5)
```

Shapes:

```text
x     (B, T, d_model)
mu    (B, T, 1)
var   (B, T, 1)
x_hat (B, T, d_model)
```

This is the core normalization structure behind LayerNorm.

Multiple axes can be reduced together:

```python
x.mean(dim=(1, 2))                  # (B,)
x.mean(dim=(1, 2), keepdim=True)    # (B, 1, 1)
```

---

## 7. `sum`, `mean`, `max`, and `argmax`

```python
token_score = x.mean(dim=-1)        # (B, T)

max_values, max_indices = token_score.max(dim=-1)
```

Shapes:

```text
token_score  (B, T)
max_values   (B,)
max_indices  (B,)
```

If only the location is needed:

```python
max_indices = token_score.argmax(dim=-1)
```

---

## 8. Matrix Multiplication

### Matrix multiplication

```python
Q = x @ W_Q
```

Shape rule:

```text
(B, T, d_model) @ (d_model, d_model)
-> (B, T, d_model)
```

The final two dimensions are treated as matrix dimensions; leading dimensions act as batch dimensions.

### Elementwise multiplication

```python
A * B
```

multiplies corresponding entries.

### Matrix multiplication

```python
A @ B
```

contracts over the shared inner dimension.

For vectors:

```python
q @ k
```

is a dot product.

---

## 9. Batched Attention Matrix Multiplication

After head splitting:

```python
Q.shape == (B, H, T, d_head)
K.shape == (B, H, T, d_head)
V.shape == (B, H, T, d_head)
```

Attention scores:

```python
scores = Q @ K.transpose(-2, -1)
scores = scores / (d_head ** 0.5)
```

Shape rule:

```text
(B, H, T, d_head)
@
(B, H, d_head, T)
->
(B, H, T, T)
```

Apply the causal mask:

```python
mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1
)

scores = scores.masked_fill(mask, float("-inf"))
```

Normalize across key positions:

```python
weights = torch.softmax(scores, dim=-1)
```

Each attention row sums to approximately 1:

```python
weights.sum(dim=-1)
```

Aggregate values:

```python
out = weights @ V
```

Shape rule:

```text
(B, H, T, T)
@
(B, H, T, d_head)
->
(B, H, T, d_head)
```

Recombine heads:

```python
out = out.transpose(1, 2)
out = out.reshape(B, T, d_model)
```

Final shape:

```text
(B, T, d_model)
```

Full shape flow:

```text
(B, T, d_model)
-> (B, T, H, d_head)
-> (B, H, T, d_head)
-> (B, H, T, T)
-> (B, H, T, d_head)
-> (B, T, H, d_head)
-> (B, T, d_model)
```

---

## 10. `torch.matmul`, `@`, `mm`, `mv`, and `dot`

For the operations used in this course:

```python
A @ B
torch.matmul(A, B)
```

are the preferred general forms.

More specialized variants:

```python
torch.dot(x, y)   # 1D dot product
torch.mv(A, x)    # 2D matrix × 1D vector
torch.mm(A, B)    # 2D matrix × 2D matrix
torch.bmm(A, B)   # batched 3D matrix multiplication
```

`@` / `torch.matmul` naturally handle the higher-dimensional tensors used in attention.

---

## 11. `einsum`

The attention score calculation:

```python
scores = Q @ K.transpose(-2, -1)
```

can also be written:

```python
scores = torch.einsum(
    "bhid,bhjd->bhij",
    Q,
    K
)
```

Interpretation:

```text
b = batch
h = head
i = query position
j = key position
d = head feature
```

Since `d` appears in the inputs but not the output, it is summed out:

```text
bhid,bhjd -> bhij
```

Similarly:

```python
out = torch.einsum(
    "bhij,bhjd->bhid",
    weights,
    V
)
```

sums over key position `j`.

Use `@` in ordinary Transformer code; use `einsum` when explicit index notation is clearer.

---

## 12. Autograd Essentials

Model parameters such as:

```python
W_Q = torch.randn(
    d_model,
    d_model,
    requires_grad=True
)
```

are leaf tensors.

Operations involving them create an autograd graph:

```python
Q = x @ W_Q
scores = Q @ K.transpose(-2, -1)
weights = torch.softmax(scores, dim=-1)
out = weights @ V
```

A scalar loss can then backpropagate through the entire graph:

```python
loss = out.pow(2).mean()
loss.backward()
```

Gradients are stored on leaf parameters:

```python
W_Q.grad
W_K.grad
W_V.grad
```

---

## 13. Gradient Accumulation

PyTorch accumulates gradients by default.

Typical training pattern:

```python
optimizer.zero_grad()

logits = model(x)
loss = loss_fn(logits, targets)

loss.backward()
optimizer.step()
```

Without `zero_grad()`, new gradients are added to existing `.grad` values.

---

## 14. `detach()` and `no_grad()`

### `detach`

```python
saved_output = out.detach()
```

The values are preserved, but the returned tensor is disconnected from the autograd graph.

Use it when a tensor should no longer participate in gradient computation.

Do not insert `detach()` into model computations unless gradient flow is intentionally being stopped.

### `torch.no_grad`

For inference or evaluation:

```python
with torch.no_grad():
    logits = model(x)
```

PyTorch does not build the autograd graph inside the block.

---

## 15. Leaf and Non-Leaf Tensors

```python
W_Q.is_leaf
```

is `True` because `W_Q` was directly created as a trainable tensor.

After:

```python
Q = x @ W_Q
```

`Q.is_leaf` is `False` because it was produced by another differentiable operation.

By default, `.grad` is retained on leaf tensors such as model parameters.

---

## 16. `.backward()` and VJPs

For a scalar loss:

```python
loss.backward()
```

reverse-mode autograd computes the gradients of the loss with respect to all connected trainable parameters.

This corresponds to a vector-Jacobian product.

For a non-scalar output, an upstream vector must be supplied:

```python
y.backward(v)
```

which computes the VJP associated with `v`.

In ordinary model training, the final loss is scalar, so:

```python
loss.backward()
```

is sufficient.

---

## 17. Debugging Shape Assertions

During implementation, encode expected mathematics directly:

```python
assert Q.shape == (B, H, T, d_head)
assert K.shape == (B, H, T, d_head)
assert V.shape == (B, H, T, d_head)

assert scores.shape == (B, H, T, T)
assert weights.shape == (B, H, T, T)

assert out.shape == (B, T, d_model)
```

A useful general habit:

> Predict the tensor shape first; run the code second.

For Transformer implementation, most tensor bugs are easier to diagnose by tracing the meaning and size of each axis than by inspecting numerical values.
