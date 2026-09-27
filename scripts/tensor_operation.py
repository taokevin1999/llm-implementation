import torch
from llm_impl.utils import set_seed

x = torch.tensor(
    [
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        [
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
        ],
    ]
)

print("x:")
print(x)

print("\nshape:", x.shape)
print("ndim:", x.ndim)
print("number of elements:", x.numel())

print("\n--- one token embedding ---")
print(x[0, 2])
print("shape:", x[0, 2].shape)

print("\n--- one whole sequence ---")
print(x[0])
print("shape:", x[0].shape)

print("\n--- token 2 from every sequence ---")
print(x[:, 2])
print("shape:", x[:, 2].shape)

print("\n--- feature 1 from every token and sequence ---")
print(x[:, :, 1])
print("shape:", x[:, :, 1].shape)

print("\n--- unsqueeze examples ---")

v = torch.tensor([10.0, 20.0, 30.0])

print("original:")
print(v)
print("shape:", v.shape)

v0 = v.unsqueeze(0)
print("\nunsqueeze(0):")
print(v0)
print("shape:", v0.shape)

v1 = v.unsqueeze(1)
print("\nunsqueeze(1):")
print(v1)
print("shape:", v1.shape)

a = x[:, 2:3, :]
b = x[:, 2, :].unsqueeze(1)

print(a.shape)
print(b.shape)
print(torch.equal(a, b))

x = torch.arange(24)

print(x)
print("shape:", x.shape)

y = x.reshape(4, 6)

print(y)
print("shape:", y.shape)

y = x.reshape(4, -1) #-1 let torch infer the value for that dimension. Only one -1 is allowed

print(y.shape)

y = x.view(4, 6) #view reinterpret with another shape but does not alter memory

print(y.shape)

# transformer type example
B = 2
T = 3
d_model = 8

x = torch.arange(B * T * d_model).reshape(B, T, d_model)

print(x.shape)

H = 2
d_head = d_model // H

x_heads = x.reshape(B, T, H, d_head)

print(x_heads.shape)

# transformer usually use (batch, head, token, head feature) so each attention head can process the entire seq (across token)
# tranpose from (B,T,H,d_h) to (B,H,T,d_h)
x_heads = x_heads.transpose(1, 2)

print(x_heads.shape)

#permute takes a complete new ordering
#equivalent to tranpose when only 2 coord changes
x_heads = x_heads.permute(3,0,1,2)

print(x_heads.shape)

#transformer data processing pipeline: (B,T,d_model)->(B,T,H,d_h)->(B,H,T,d_h)
#Q,K,V will all be (B,H,T,d_h)
#QK^T will be (B,H,T,d_h) @ (B,H,d_h,T) -> (B,H,T,T)
B = 2
T = 3
d_model = 8
H = 2
d_head = d_model // H

x = torch.arange(B * T * d_model).reshape(B, T, d_model)

print("original:", x.shape)

x = x.reshape(B, T, H, d_head)
print("split heads:", x.shape)

x = x.transpose(1, 2)
print("reorder heads:", x.shape)

# transpose and contiguity

x = torch.arange(12).reshape(3, 4)

print(x)
print("shape:", x.shape)
print("is contiguous:", x.is_contiguous())

y = x.transpose(0, 1)

print(y)
print("shape:", y.shape)
print("is contiguous:", y.is_contiguous())

# stride shows how data is "walked"
# view will not work for transposed tensor
print("x stride:", x.stride())
print("y stride:", y.stride())

# .contiguous() make a tensor contiguous
y_contiguous = y.contiguous()
print(y_contiguous.is_contiguous())

# transformer split head -> transpose -> transpose back -> merge head
B, T, d_model = 2, 3, 8
H = 2
d_head = d_model // H

x = torch.randn(B, T, d_model)

print("original:", x.shape)

x = x.reshape(B, T, H, d_head)
print("split:", x.shape)

x = x.transpose(1, 2)
print("heads forward:", x.shape)
print("contiguous:", x.is_contiguous())

x = x.transpose(1, 2)
print("heads back:", x.shape)

x = x.reshape(B, T, d_model)
print("merged:", x.shape)

# Broadcasting lets PyTorch combine tensors with different shapes without manually copying the smaller tensor.
x = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)

b = torch.tensor([10.0, 20.0, 30.0])

print("x shape:", x.shape)
print("b shape:", b.shape)

y = x + b

print(y)
print("y shape:", y.shape)

# torch compare dimensions from right to left, and dimensions are compatible if they are equal or if one of them is 1.
# missing dimensions are treated as 1.

#3D tensor example
x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

b = torch.tensor([100.0, 200.0, 300.0])

y = x + b

print("x shape:", x.shape)
print("b shape:", b.shape)
print("y shape:", y.shape)
print(y)

#broadcasting to coord besides last
x = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)

c = torch.tensor([[10.0], [20.0]])

print("x shape:", x.shape)
print("c shape:", c.shape)

print(x + c)

#use unsqueeze with broadcast to make right end dimensions match
scale = torch.tensor([1.0, 2.0])
x = torch.randn(2, 4, 3)
scale = scale.unsqueeze(1).unsqueeze(2)
print(scale.shape)

y = x * scale

#Causal mask example: S = QK^T has dim (B,H,T,T)
scores = torch.randn(2, 4, 5, 5)

mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))

print("scores:", scores.shape)
print("mask:", mask.shape)
scores = scores.masked_fill(~mask, float("-inf"))
print('scores:',scores)

# subscripting with None create new dimension like unsequeeze
scale[:, None, None].shape

# functions such as mean, var, std,... take the form x.mean(dim=..,keepdim=False)
# dim = k, removes the kth dimension when computing stats
# eg: x.shape = (4,6,2) -> x.mean(dim=1).shape = (4,2)

# layernorm example
x = torch.randn(2, 4, 3)

mu = x.mean(dim=-1, keepdim=True) #keepdim allows easy broadcasting

print("x:", x.shape)
print("mu:", mu.shape)

centered = x - mu
var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
eps = 1e-5
x_hat = (x - mu) / torch.sqrt(var + eps)

gamma = torch.ones(3)
beta = torch.zeros(3)

y = gamma * x_hat + beta

# verify output mean and var
print(x_hat.mean(dim=-1))
print(x_hat.var(dim=-1, unbiased=False))

# torch can reduce over multiple dimension
x = torch.randn(2, 4, 3)

y = x.mean(dim=(1, 2))

# torch max also returns position of the maximum entry
# argmax for only position
x = torch.tensor([
    [1.0, 5.0, 2.0],
    [8.0, 3.0, 4.0],
])

values , indices = x.max(dim=1,keepdim=True)

print(values)
print(indices)

# softmax is also dimension reduction operation
scores = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 0.0],
])
probs = torch.softmax(scores, dim=-1)

# For S with dimension (B,H,T,T), attention is obtained by applying softmax to last dim
# first T: all query i
# second T: all key j

# Batch matrix multiplication (B,H,T,d_h) @ (B,H,d_h,T) -> (B,H,T,T) for QK^T in each batch and each head
B = 2
H = 4
T = 5
d_head = 8

Q = torch.randn(B, H, T, d_head)
K = torch.randn(B, H, T, d_head)
V = torch.randn(B, H, T, d_head)

K_t = K.transpose(-2, -1)
print(K_t.shape)

scores = Q @ K_t # all B*H matrix mult are done in parallel. torch only does matrix mult for last 2 dims
print(scores.shape)

scores = (Q @ K.transpose(-2, -1)) / (d_head ** 0.5) # scaling by root(d_h)
scores = (Q @ K.transpose(-2, -1)) / (d_head ** 0.5)

mask = torch.tril(torch.ones(T, T, dtype=torch.bool)) #add mask

scores = scores.masked_fill(~mask, float("-inf"))
print('scores:',scores)

weights = torch.softmax(scores, dim=-1) #apply softmax to get attention matrices A
out = weights @ V # perform AV for each batch and head

print(out.shape)

#different mult operations
A = torch.randn(4, 3)
B = torch.randn(3, 4)
x = torch.randn(3)

y = torch.mv(A, x) #matrix-vector mult
C = torch.mm(A, B) #matrix-matrix mult

#matrix mult broadcast when last 2 dims match for regular matrix mult
A = torch.randn(10, 4, 3)
B = torch.randn(3, 5)

C = A @ B # (10,4,5)

# einsum allows user to specify coordiante that mult and sum is done over
# another formulation of QK^T
scores = torch.einsum(
    "bhid,bhjd->bhij",
    Q,
    K
) # d appear in both input, but not output so it is being summed over

#tensor and grad
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x * 2
z = y.sum()

print(x.is_leaf) #True; x was created directly and specified require_grad=T
print(y.is_leaf) #False; came from another differentiable operation

z.backward() #compute grad via backprop

print(x.grad) #som tensor; torch store grad info in leaf tensor
print(y.grad) #None; torch usually does not store grad info in non leaf

#detach creates a tensor that share same numerical value but removed from the autograd graph
x = torch.tensor([2.0], requires_grad=True)

y = x * 3
z = y.detach()
print(y.requires_grad)
print(z.requires_grad) # False; due to detach

#for a scalar loss: .backward() computes all d_L/d_theta
#for a vector v(x): v.backward(a) need to specify a vector direction a for a^T @ J where J is jacobian

# an attention example run through
set_seed(42)

B = 2
T = 5
d_model = 12
H = 3

assert d_model % H == 0

d_head = d_model // H

x = torch.randn(B, T, d_model) #before splitting heads
print("x:", x.shape)

#weight for Q K V
W_Q = torch.randn(d_model, d_model, requires_grad=True)
W_K = torch.randn(d_model, d_model, requires_grad=True)
W_V = torch.randn(d_model, d_model, requires_grad=True)

Q = x @ W_Q
K = x @ W_K
V = x @ W_V

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

#splitting heads
Q = Q.reshape(B, T, H, d_head)
K = K.reshape(B, T, H, d_head)
V = V.reshape(B, T, H, d_head)

#flip dimension to (B,H,T,d_h)
Q = Q.transpose(1, 2)
K = K.transpose(1, 2)
V = V.transpose(1, 2)

print("Q split:", Q.shape)
print("K split:", K.shape)
print("V split:", V.shape)

#compute attention matrices
scores = Q @ K.transpose(-2, -1)
print("scores:", scores.shape)
scores = scores / (d_head ** 0.5)

#compute mask
mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1
)

print(mask)
scores = scores.masked_fill(mask, float("-inf"))

#apply softmax
weights = torch.softmax(scores, dim=-1)
print("weights:", weights.shape)
print("weights:")
print(weights)

#check rows are probabilities summing to one
row_sums = weights.sum(dim=-1)
print(row_sums)

#compute AV
out = weights @ V
print("attention output:", out.shape)

#undo transposition and merge heads
out = out.transpose(1, 2)
out = out.reshape(B, T, d_model)

#example loss for autograd
loss = out.pow(2).mean()
loss.backward()
print("W_Q grad:", W_Q.grad.shape)
print("W_K grad:", W_K.grad.shape)
print("W_V grad:", W_V.grad.shape)