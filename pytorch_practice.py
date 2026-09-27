import torch
import torch.nn as nn
torch.manual_seed(42)

#region
# ============================================================
# Assignment 1: Tensor basics
# ============================================================

# 1. Create a 1D floating-point tensor containing:
#    1.0, 2.0, 3.0, 4.0
x = torch.tensor([1.0,2.0,3.0,4.0])


# 2. Create a 2D floating-point tensor:
#
#    [1.0, 2.0, 3.0]
#    [4.0, 5.0, 6.0]
#
X = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])


# 3. Print x and X with descriptive labels.
print('x:',x)
print('X:',X)
# 4. Print the shape of x and X.
print('x shape:',x.shape)
print('X shape:',X.shape)

# 5. Print the dtype of X.
print('X dtype:',X.dtype)

# 6. Extract the number 5.0 from X and store it as:
value = X[1][1]


# 7. Print value with a descriptive label.
print('value:',value)
#endregion

#region
# ============================================================
# Assignment 2: Indexing and tensor arithmetic
# ============================================================

A = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

B = torch.tensor([
    [10.0, 20.0, 30.0],
    [40.0, 50.0, 60.0]
])


# 1. Extract the first row of A.
first_row = A[0,:]
print('first row of A:',first_row)

# 2. Extract the second column of A.
second_column = A[:,1]
print('second column of A:',second_column)

# 3. Compute elementwise addition of A and B.
C = A+B
print('A+B:',C)

# 4. Compute elementwise multiplication of A and B.
D = A*B
print('A*B:',D)

# 5. Multiply every element of A by 2.
E = A*2
print('A*2:',E)


# 6. Compute the sum of ALL entries of A.
total = torch.sum(A)
print('sum of A:',total)


# 7. Print the results with descriptive labels.
#endregion

#region
# ============================================================
# Assignment 3: Matrix multiplication
# ============================================================

X = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

W = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])


# 1. Print the shapes of X and W.
print('X shape:',X.shape)
print('W shape:',W.shape)

# 2. Compute matrix multiplication XW using the @ operator.
Y = X @ W
print('Y:',Y)


# 3. Print Y.


# 4. Print the shape of Y.
print('Y shape:',Y.shape)

# 5. Compute elementwise multiplication between
#    the first row of X and the first column of W.
products = X[0,:]*W[:,0]

# 6. Sum those products.
manual_entry = sum(products)


# 7. Extract Y[0, 0].
matrix_entry = Y[0,0]


# 8. Print manual_entry and matrix_entry.
print('manual entry:', manual_entry)
print('matrix entry:', matrix_entry)
#endregion

#region
# ============================================================
# Assignment 4: Bias and broadcasting
# ============================================================

X = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

W = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])

b = torch.tensor([1.0, -1.0])


# 1. Compute the matrix product.
linear_output = X @ W


# 2. Add the bias vector.
Z = linear_output + b


# 3. Print the shape of linear_output.
print('linear output shape:',linear_output.shape)

# 4. Print the shape of b.
print('shape of b:',b.shape)

# 5. Print Z.
print('Z:',Z)

# 6. Manually construct a 2x2 tensor in which
#    the bias vector b has been repeated twice.
#
#    Do NOT use torch.repeat yet.
#
B_manual = torch.stack([b, b])


# 7. Verify manually that adding B_manual gives
#    the same result as broadcasting.
Z_manual = linear_output+B_manual


# 8. Print Z_manual.
print('Z_manual:',Z_manual)

# 9. Compare Z and Z_manual using:
same_result = torch.allclose(Z, Z_manual)

# 10. Print same_result.
print('same_result:',same_result)
#endregion

#region
# ============================================================
# Assignment 5: Handwritten ReLU
# ============================================================

Z = torch.tensor([
    [-2.0, -0.5, 0.0],
    [1.0,  3.0, -4.0]
])


# 1. Create a Boolean tensor indicating where Z > 0.
positive_mask = Z>0


# 2. Print the mask and its dtype.
print('positive mask:')
print(positive_mask)
print('positive mask dtype:')
print(positive_mask.dtype)

# 3. Create a tensor of zeros having exactly the
#    same shape and dtype as Z.
zeros = torch.zeros(2,3)


# 4. Use torch.where to construct ReLU(Z):
#
#       Z[i,j] if Z[i,j] > 0
#       0      otherwise
#
relu_output = positive_mask*Z


# 5. Print the result.
print('RELU output')
print(relu_output)

# 6. Put the same logic into a Python function.
def relu(x):
    mask = x>0
    return(x*mask)


# 7. Apply your function to Z.
function_output = relu(Z)


# 8. Check that the two implementations agree.
same_result = torch.allclose(relu_output, function_output)


# 9. Print the result of the check.
print('same check:',same_result)

#endregion

#region
# ============================================================
# Assignment 6: Reshaping and reductions
# ============================================================

x = torch.tensor([
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
])


# 1. Print the shape of x.
print('shape of x:',x.shape)

# 2. Reshape x into a 2x3 matrix.
X = x.reshape(2,3)


# 3. Print X and its shape.
print('X:')
print(X)
print('X shape:',X.shape)

# 4. Compute the sum of ALL entries in X.
total_sum = torch.sum(X)
print('total sum:',total_sum)

# 5. Compute the sum of each ROW.
row_sums = torch.sum(X,1)
print('row sums:',row_sums)

# 6. Compute the sum of each COLUMN.
column_sums = torch.sum(X,0)
print('column sums:',column_sums)

# 7. Compute the mean of ALL entries.
overall_mean = torch.mean(X)
print('overall mean:',overall_mean)


# 8. Compute the mean of each COLUMN.
column_means = torch.mean(X,0)
print('column means:',column_means)

# 9. Print all results with descriptive labels.

#endregion

#region
# ============================================================
# Assignment 7: Handwritten mean squared error
# ============================================================

y_true = torch.tensor([3.0, 5.0, 5.0, 7.0])
y_pred = torch.tensor([2.0, 4.0, 5.0, 8.0])


# 1. Compute the prediction errors:
#
#       y_pred - y_true
#
errors = y_pred - y_true
print('errors:',errors)

# 2. Square each error.
squared_errors = errors**2
print('squared errors:',squared_errors)

# 3. Compute the mean squared error.
mse = torch.mean(squared_errors)
print('MSE:',mse)

# 4. Write your own reusable MSE function.
def mean_squared_error(y_pred, y_true):
    err = y_pred - y_true
    return(torch.mean(err**2))


# 5. Compute the loss using your function.
function_mse = mean_squared_error(y_pred, y_true)


# 6. Check whether the direct calculation
#    and function calculation agree.
same_result = torch.allclose(function_mse,mse)
print('same result:',same_result)

# 7. Print all relevant results.

#endregion

#region
# ============================================================
# Assignment 8: Manual gradient of MSE
# ============================================================

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor(1.0)


# 1. Compute predictions y_pred = w*x.
y_pred = w*x


# 2. Compute the MSE using your function from Assignment 7.
loss = mean_squared_error(y_pred,y_true)

# 3. Compute the errors.
errors = y_pred-y_true


# 4. Compute each observation's contribution to dL/dw:
#
#       2 * error_i * x_i
#
gradient_terms = 2*errors*x


# 5. Take the mean to obtain dL/dw.
dL_dw = torch.mean(gradient_terms)


# 6. Print predictions, loss, gradient terms, and dL/dw.
print('pred:',y_pred)
print('loss:',loss)
print('grad terms:',gradient_terms)
print('dL/dw:',dL_dw)

#endregion

#region
# ============================================================
# Assignment 9: One manual gradient-descent step
# ============================================================

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor(1.0)
learning_rate = 0.1


# 1. Compute predictions using the current w.
y_pred = w*x


# 2. Compute the current loss.
loss_before = mean_squared_error(y_pred,y_true)


# 3. Compute the errors.
errors = y_pred-y_true


# 4. Compute dL/dw manually.
dL_dw = torch.mean(2*errors*x)


# 5. Update w using gradient descent.
w_new = w-learning_rate*dL_dw


# 6. Compute new predictions using w_new.
y_pred_new = w_new*x


# 7. Compute the new loss.
loss_after = mean_squared_error(y_pred_new,y_true)


# 8. Print all important quantities.
print('loss before:',loss)
print('grad:',dL_dw)
print('loss after:',loss_after)


#endregion

#region
# ============================================================
# Assignment 10: Manual training loop
# ============================================================

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

w = torch.tensor(0.0)
learning_rate = 0.1
num_steps = 10


for step in range(num_steps):

    # 1. Forward pass
    y_pred = w*x

    # 2. Compute loss
    loss = mean_squared_error(y_pred,y_true)

    # 3. Compute errors
    errors = y_pred-y_true

    # 4. Compute manual gradient dL/dw
    dL_dw = torch.mean(2*x*errors)

    # 5. Update w
    w = w-learning_rate*dL_dw

    # 6. Print training progress
    print(
    f"step: {step:2d} | "
    f"loss: {loss.item():8.4f} | "
    f"w: {w.item():.4f}"
)

#endregion

#region
# ============================================================
# Assignment 11: Train weight and bias manually
# ============================================================

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([3.0, 5.0, 7.0])

w = torch.tensor(0.0)
b = torch.tensor(0.0)

learning_rate = 0.05
num_steps = 50


for step in range(num_steps):

    # 1. Forward pass
    y_pred = w*x+b

    # 2. Compute MSE
    loss = mean_squared_error(y_pred,y_true)

    # 3. Compute errors
    errors = y_pred-y_true

    # 4. Compute dL/dw
    #
    # dL/dw = mean(2 * error_i * x_i)
    #
    dL_dw = torch.mean(2*errors*x)

    # 5. Compute dL/db
    #
    # dL/db = mean(2 * error_i)
    #
    dL_db = torch.mean(2*errors)

    # 6. Update both parameters
    w = w-learning_rate*dL_dw
    b = b-learning_rate*dL_db

    # 7. Print every 5 steps
    if step % 5 == 0:
        print(
    f"step: {step:2d} | "
    f"loss: {loss.item():8.4f} | "
    f"w: {w.item():.4f} | "
    f"b: {b.item():.4f}"
)
#endregion

#region
# ============================================================
# Assignment 12: Linear model with multiple input features
# ============================================================

X = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 1.0]
])

y_true = torch.tensor([
    [2.0],
    [-1.0],
    [1.0],
    [3.0]
])

W = torch.tensor([
    [0.0],
    [0.0]
])

b = torch.tensor(0.0)

learning_rate = 0.1
num_steps = 50
n = X.shape[0]

for step in range(num_steps):

    # 1. Forward pass:
    #
    # y_pred = XW + b
    #
    y_pred = X @ W + b

    # 2. Compute MSE.
    loss = mean_squared_error(y_pred,y_true)

    # 3. Compute errors.
    errors = y_pred-y_true

    # 4. Compute gradient with respect to W.
    #
    # Hint:
    #
    # dL/dW = (2/n) X^T errors
    #
    dL_dW = 2 * X.T @ errors/n

    # 5. Compute gradient with respect to b.
    dL_db = torch.mean(2*errors)

    # 6. Update parameters.
    W = W-learning_rate*dL_dW
    b = b-learning_rate*dL_db

    # 7. Print every 10 steps.
    if step % 10 == 0:
        print(
            f"step: {step:2d} | "
            f"loss: {loss.item():8.4f} | "
            f"b: {b.item():.4f}")
        print("W:")
        print(W)
#endregion

#region
# ============================================================
# Assignment 13: Tiny FNN forward pass
# ============================================================

X = torch.tensor([
    [1.0, 2.0],
    [3.0, 1.0],
    [-1.0, 2.0]
])

W1 = torch.tensor([
    [1.0, -1.0],
    [2.0,  1.0]
])

b1 = torch.tensor([0.0, 1.0])

W2 = torch.tensor([
    [1.0],
    [2.0]
])

b2 = torch.tensor(0.5)

y_true = torch.tensor([
    [10.0],
    [6.0],
    [11.0]
])

# 1. Compute first-layer pre-activation:
Z1 = X @ W1 +b1


# 2. Apply your handwritten ReLU function from Assignment 5:
H = relu(Z1)


# 3. Compute output-layer prediction:
y_pred = H @ W2 + b2

# 4. Compute prediction errors
errors = y_pred-y_true

# 5. Compute handwritten MSE
loss = mean_squared_error(y_pred,y_true)

#6. Print shapes:
# X, W1, Z1, H, W2, y_pred
print('X shape:',X.shape)
print('W1 shape:',W1.shape)
print('Z1 shape:',Z1.shape)
print('H shape:',H.shape)
print('W2 shape:',W2.shape)
print('y pred shape:',y_pred.shape)

# 7. Print Z1, H, and y_pred, errors and loss.
print('Z1:',Z1)
print('H:',H)
print('y pred:', y_pred)
print('error:',errors)
print('loss:',loss)

#endregion

#region
# ============================================================
# Assignment 15: Output-layer gradients
# ============================================================

n = y_true.shape[0]


# 1. Compute dL/dy_pred
dL_dy = 2*errors/n


# 2. Compute dL/dW2
dL_dW2 = H.T @ dL_dy


# 3. Compute dL/db2
dL_db2 = torch.sum(dL_dy)


# 4. Print all three gradients and their shapes.
print('dL_dy shape:',dL_dy.shape)
print('dL_dy:',dL_dy)
print('dL_dW2 shape:',dL_dW2.shape)
print('dL_dW2 :',dL_dW2)
print('dL_db2 shape:',dL_db2.shape)
print('dL_db2:',dL_db2)
#endregion

#region
# ============================================================
# Assignment 16: Backprop through output layer and ReLU
# ============================================================

# 1. Propagate the gradient back to H.
dL_dH = dL_dy @ W2.T


# 2. Create the ReLU derivative mask.
relu_grad = Z1>0


# 3. Propagate through ReLU.
dL_dZ1 = dL_dH * relu_grad


# 4. Print values and shapes.
print('dL_dH shape:',dL_dH.shape)
print('relu_grad shape:',relu_grad.shape)
print('dL_dZ1 shape:',dL_dZ1.shape)
print('dL_dH:',dL_dH)
print('relu_grad:',relu_grad)
print('dL_dZ1:',dL_dZ1)

#endregion

#region

# ============================================================
# Assignment 17: First-layer gradients
# ============================================================

# 1. Compute dL/dW1.
dL_dW1 = X.T @ dL_dZ1


# 2. Compute dL/db1 by summing over the batch dimension.
dL_db1 = torch.sum(dL_dZ1,0)


# 3. Print shapes.
print("dL_dW1 shape:", dL_dW1.shape)
print("dL_db1 shape:", dL_db1.shape)


# 4. Print gradients.
print("dL_dW1:")
print(dL_dW1)

print("dL_db1:")
print(dL_db1)

#endregion

#region full FNN training from scratch

# ============================================================
# Assignment 18: Manual FNN training loop
# ============================================================

X = torch.tensor([
    [1.0, 2.0],
    [3.0, 1.0],
    [-1.0, 2.0]
])

y_true = torch.tensor([
    [10.0],
    [6.0],
    [11.0]
])

W1 = torch.tensor([
    [1.0, -1.0],
    [2.0,  1.0]
])

b1 = torch.tensor([0.0, 1.0])

W2 = torch.tensor([
    [1.0],
    [2.0]
])

b2 = torch.tensor(0.5)

learning_rate = 0.01
num_steps = 101


for step in range(num_steps):

    # ========================================
    # Forward pass
    # ========================================

    Z1 = X @ W1 + b1

    H = relu(Z1)

    y_pred = H @ W2 + b2

    loss = mean_squared_error(y_pred,y_true)
    errors = y_pred-y_true

    # ========================================
    # Backward pass
    # ========================================

    n = y_true.shape[0]

    # dL / dy_pred
    dL_dy = 2*errors/n

    # output layer
    dL_dW2 = H.T @ dL_dy
    dL_db2 = torch.sum(dL_dy)

    # propagate to hidden layer
    dL_dH = dL_dy @ W2.T

    # through ReLU
    relu_grad = Z1>0
    dL_dZ1 = dL_dH * relu_grad

    # first layer
    dL_dW1 = X.T @ dL_dZ1
    dL_db1 = torch.sum(dL_dZ1,0)


    # ========================================
    # Parameter update
    # ========================================

    W1 = W1-learning_rate*dL_dW1
    b1 = b1-learning_rate*dL_db1

    W2 = W2-learning_rate*dL_dW2
    b2 = b2-learning_rate*dL_db2


    # ========================================
    # Training diagnostics
    # ========================================

    if step % 20 == 0:
        print(
            f"step: {step:3d} | "
            f"loss: {loss.item():.6f}"
        )

print("final predictions:")
print(y_pred)

print("targets:")
print(y_true)

print("W1:")
print(W1)

print("b1:")
print(b1)

print("W2:")
print(W2)

print("b2:", b2)

#endregion

#region Full training with autograd
# ============================================================
# Assignment 19: Verify manual gradients with autograd
# ============================================================

X = torch.tensor([
    [1.0, 2.0],
    [3.0, 1.0],
    [-1.0, 2.0]
])

y_true = torch.tensor([
    [10.0],
    [6.0],
    [11.0]
])


W1 = torch.tensor([
    [1.0, -1.0],
    [2.0,  1.0]
], requires_grad=True)

b1 = torch.tensor(
    [0.0, 1.0],
    requires_grad=True
)

W2 = torch.tensor([
    [1.0],
    [2.0]
], requires_grad=True)

b2 = torch.tensor(
    0.5,
    requires_grad=True
)


# ========================================
# Forward pass
# ========================================

Z1 = X @ W1 + b1
H = relu(Z1)
y_pred = H @ W2 + b2

loss = mean_squared_error(y_pred,y_true)


# ========================================
# Let PyTorch perform backpropagation
# ========================================

loss.backward()


# ========================================
# Inspect gradients
# ========================================

print("W1 gradient:")
print(W1.grad)

print("b1 gradient:")
print(b1.grad)

print("W2 gradient:")
print(W2.grad)

print("b2 gradient:")
print(b2.grad)

#endregion

#region
# ============================================================
# Assignment 20: Autograd + manual parameter updates
# ============================================================

X = torch.tensor([
    [1.0, 2.0],
    [3.0, 1.0],
    [-1.0, 2.0]
])

y_true = torch.tensor([
    [10.0],
    [6.0],
    [11.0]
])

W1 = torch.tensor([
    [1.0, -1.0],
    [2.0,  1.0]
], requires_grad=True)

b1 = torch.tensor(
    [0.0, 1.0],
    requires_grad=True
)

W2 = torch.tensor([
    [1.0],
    [2.0]
], requires_grad=True)

b2 = torch.tensor(
    0.5,
    requires_grad=True
)

learning_rate = 0.01
num_steps = 101


for step in range(num_steps):

    # --------------------
    # Forward
    # --------------------

    Z1 = X @ W1 + b1
    H = relu(Z1)
    y_pred = H @ W2 + b2
    loss = mean_squared_error(y_pred,y_true)


    # --------------------
    # Backward
    # --------------------

    loss.backward()


    # --------------------
    # Parameter update
    # --------------------

    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad


    # --------------------
    # Clear old gradients
    # --------------------

    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()


    # --------------------
    # Diagnostics
    # --------------------

    if step % 20 == 0:
        print(
            f"step: {step:3d} | "
            f"loss: {loss.item():.6f}"
        )
#endregion

#region using nn.linear
# ============================================================
# Assignment 21: nn.Linear
# ============================================================

X = torch.tensor([
    [1.0, 2.0],
    [3.0, 1.0],
    [-1.0, 2.0]
])

y_true = torch.tensor([
    [10.0],
    [6.0],
    [11.0]
])


# 1. Create a linear layer:
#
#    2 input features -> 2 hidden features
#
layer1 = nn.Linear(2,2)


# 2. Create the output layer:
#
#    2 hidden features -> 1 output feature
#
layer2 = nn.Linear(2,1)


# 3. Print the layers.
print("layer1:", layer1)
print("layer2:", layer2)


# 4. Print the shapes of their weights and biases.
print("layer1 weight shape:", layer1.weight.shape)
print("layer1 bias shape:", layer1.bias.shape)

print("layer2 weight shape:", layer2.weight.shape)
print("layer2 bias shape:", layer2.bias.shape)


# 5. Compute a forward pass.
Z1 = layer1(X)
H = relu(Z1)
y_pred = layer2(H)


# 6. Compute your handwritten MSE.
loss = mean_squared_error(y_pred,y_true)


print("y_pred:")
print(y_pred)

print("loss:", loss)

#endregion

#region package nn.Linear into class
# ============================================================
# Assignment 22: First nn.Module
# ============================================================

class SimpleFNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 2 inputs -> 2 hidden units
        self.layer1 = nn.Linear(2,2)

        # 2 hidden units -> 1 output
        self.layer2 = nn.Linear(2,1)

        self.relu = nn.ReLU()

    def forward(self, x):

        z1 = self.layer1(x)

        h = self.relu(z1)

        y_pred = self.layer2(z1)

        return y_pred
model = SimpleFNN()
print(model)

y_pred = model(X)

loss = mean_squared_error(y_pred, y_true)

print("y_pred:")
print(y_pred)

print("loss:", loss)

#endregion

#region nn.MSEloss
# ============================================================
# Assignment 23: nn.MSELoss
# ============================================================

X = torch.tensor([
    [1.0, 2.0],
    [3.0, 1.0],
    [-1.0, 2.0]
])

y_true = torch.tensor([
    [10.0],
    [6.0],
    [11.0]
])


model = SimpleFNN()


# 1. Create PyTorch's mean squared error loss object.
criterion = nn.MSELoss()


# 2. Run the model.
y_pred = model(X)


# 3. Compute MSE using your handwritten function.
manual_loss = mean_squared_error(y_pred,y_true)


# 4. Compute MSE using nn.MSELoss.
torch_loss = criterion(y_pred,y_true)


# 5. Check whether they agree.
same_result = torch.isclose(torch_loss,manual_loss)


# 6. Print predictions and both losses.

print("y_pred:")
print(y_pred)

print("manual loss:", manual_loss)
print("torch loss:", torch_loss)
print("same result:", same_result)

#endregion

#region torch.optim.SGD
# ============================================================
# Assignment 24: torch.optim.SGD
# ============================================================

X = torch.tensor([
    [1.0, 2.0],
    [3.0, 1.0],
    [-1.0, 2.0]
])

y_true = torch.tensor([
    [10.0],
    [6.0],
    [11.0]
])

model = SimpleFNN()

criterion = nn.MSELoss()

learning_rate = 0.01


# 1. Create an SGD optimizer that will update
#    all trainable parameters in model.
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


# 2. Print the model parameters and their shapes.
for name, parameter in model.named_parameters():
    print(name, parameter.shape)

num_steps = 1001

for step in range(num_steps):

    # 1. Forward pass
    y_pred = model(X)

    # 2. Loss
    loss = criterion(y_pred,y_true)

    # 3. Clear gradients from the previous iteration
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Update parameters
    optimizer.step()

    if step % 200 == 0:
        print(
            f"step: {step:3d} | "
            f"loss: {loss.item():.6f}"
        )

with torch.no_grad():
    final_predictions = model(X)

print("final predictions:")
print(final_predictions)

print("targets:")
print(y_true)

#endregion
#region how model is stored

# ============================================================
# Assignment 26: Inspect model parameters and state_dict
# ============================================================

# 1. Print all named parameters.
for name, parameter in model.named_parameters():
    print(name,parameter)
    

# 2. Print the model's state dictionary.
print("\nstate_dict:")
print(model.state_dict())


# 3. Access individual parameters directly.
print("\nlayer1 weights:")
print(model.layer1.weight)

print("\nlayer1 bias:")
print(model.layer1.bias)

print("\nlayer2 weights:")
print(model.layer2.weight)

print("\nlayer2 bias:")
print(model.layer2.bias)

#endregion

#region saving and loading model
# ============================================================
# Assignment 27: Save and reload model state
# ============================================================

model_path = "simple_fnn_state.pt"


# 1. Save the trained model's state_dict.
torch.save(model.state_dict(),model_path)


# 2. Create a completely new model.
new_model = SimpleFNN()


# 3. Print predictions from the new model BEFORE loading.
with torch.no_grad():
    predictions_before_load = new_model(X)

print("predictions before loading:")
print(predictions_before_load)


# 4. Load the saved state dictionary from disk.
saved_state = torch.load(model_path)


# 5. Load those parameters into new_model.
new_model.load_state_dict(saved_state)


# 6. Compute predictions AFTER loading.
with torch.no_grad():
    predictions_after_load = new_model(X)


# 7. Compute predictions from the original trained model.
with torch.no_grad():
    original_predictions = model(X)


# 8. Compare them.
same_predictions = torch.allclose(predictions_after_load,original_predictions)


print("\noriginal predictions:")
print(original_predictions)

print("\npredictions after loading:")
print(predictions_after_load)

print("\nsame predictions:", same_predictions)
#endregion

#region evel mode
# ============================================================
# Assignment 28: train mode and eval mode
# ============================================================

model.train()

print("training mode:", model.training)


model.eval()

print("training mode:", model.training)


with torch.no_grad():
    predictions = model(X)

print("predictions in eval mode:")
print(predictions)