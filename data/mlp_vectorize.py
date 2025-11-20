# Multi-Layer Perceptron Implementation
import jax
import jax.numpy as jnp

learning_rate = 0.5
EPOCHS = 10000
hidden_size = 4
hidden_layers = 1

# init weights and bias
def init_wb(size_, seed=42):
    key = jax.random.PRNGKey(seed) # [0 42]
    W, B = [], []

    # In SLP, we used only one scaling factor so here we are introducing new method, 
    # using different scaling values for weights and bias
    w_scaling_factor = 0.5 # ()
    b_scaling_factor = 0.1 # ()

    for i in range(len(size_) - 1): # len(size_) = 3
        # Split into 3 - one main key and 2 for weights and bias in each layer
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # layer_size = [2, 4, 1]
        # len(layer_size) = 3
        # len(layer_size) - 1 = 3 -1 = 2
        # range(len(layer_size)-1) = range(0, 2)
        # layer_size[0] = 2
        # layer_size[1] = 4
        # layer_size[2] = 1
        #
        w = jax.random.normal(subkey1, (size_[i], size_[i+1])) * w_scaling_factor
        b = jax.random.normal(subkey2, (size_[i+1], )) * b_scaling_factor
        W.append(w)
        B.append(b)
    # len(W) and len(B) should be 2 as range(0, 2) only gives two values
    # W[0] shape should be (2, 4) as (input_size, hidden_size)
    # W[1] shape should be (4, 1) as (hidden_size, output_size)
    #
    # B[0] shape should be (4, ) as (hidden_size)
    # B[1] shape should be (1, ) as (output_size)

    return W, B

# activation functions
def fsigmoid(x): 
    # Both fsigmoid(x) and frelu(x) preserve the shape of x
    # input x: (n,)
    # output f(x): (n,)
    return 1.0 / (1.0 + jnp.exp(-x)) 

def frelu(x):
    # Both fsigmoid(x) and frelu(x) preserve the shape of x
    # input x: (n,)
    # output f(x): (n,)
    return jnp.maximum(0, x)

# forward pass
def forward_pass(X, W, B):
    Z = [] # pre-activation
    A = [X] # initialise with input data

    # len(W) should be 2
    for i in range(len(W)): # range(0,2) = 0,1
        # A stores activations of each layer; A[-1] is the latest activation.
        # We use the most recent activation as input to the next layer.
        # For layer i:
        #     z = A[-1] @ W[i] + B[i]
        #
        # z = A[-1] @ W[i] + B[i]
        # For layer 0: (2,) @ (2,4) + (4,) -> (4,)
        # For layer 1: (4,) @ (4,1) + (1,) -> (1,)
        # 
        z = jnp.dot(A[-1], W[i]) + B[i]
        
        # Relu if not output layer
        # if i=0, i < len(W) -1 -> 0 < 1 -> true then Relu
        # if i=0, i < len(W) -1 -> 1 < 1 -> false then Sigmoid
        #
        # For any layer i,
        # z[i] (pre-activation) and a[i] (post-activation) ALWAYS have the same shape,
        # because the activation function (ReLU, sigmoid, etc.) is applied elementwise. 
        #
        # Shapes of 'a' after each loop:
        # i = 0  -> hidden layer activation shape = (4,)
        # i = 1  -> output layer activation shape = (1,)
        a = frelu(z) if i < len(W) -1 else fsigmoid(z)
        # 
        Z.append(z)
        A.append(a)
    
    return Z, A

# backward pass
def backward_pass(X, y, W, B, Z, A):
    n_layers = len(W) # 2
    batch_size = X.shape[0] # 4
    dW = [None] * n_layers # [None, None]
    dB = [None] * n_layers # [None, None]

    # outer layer error term
    # Delta (δ) derivation for output layer
    #
    # Delta is the gradient of the loss wrt pre-activation z:
    # δ = ∂L/∂z
    #
    # Forward path:
    # W → z → a → L
    # Backward path:
    # L → a → z → W
    #
    # Chain rule for delta:
    # δ = ( ∂L/∂a ) * ( ∂a/∂z )
    #
    # Step 1: Loss derivative
    # Loss function (MSE without 1/2 term):
    # L = (a - y)^2
    #
    # Derivative:
    # ∂L/∂a = 2 * (a - y)
    #
    # Note: We omit the 1/2 term because it only cancels the 2 in the derivative.
    #       Keeping it simple does not change learning behavior, only the scale.
    #
    # Step 2: Sigmoid derivative
    # Activation:
    # a = sigmoid(z) = 1 / (1 + exp(-z))
    #
    # Derivative of sigmoid:
    # ∂a/∂z = a * (1 - a)
    #       = sigmoid(z) * (1 - sigmoid(z))
    #
    # Step 3: Combine using chain rule
    # δ = (2 * (a - y)) * (a * (1 - a))
    #
    # Final delta formula:
    # delta = 2 * (A[-1] - y) * (A[-1] * (1 - A[-1]))
    delta = 2 * (A[-1] - y) * (A[-1] * (1 - A[-1]))

    # Backpropagate through layers
    # Loop backward through layers: i = n_layers-1, n_layers-2, ..., 0
    # This allows backpropagation from output layer to input layer.
    for i in range(n_layers -1, -1, -1): # range(2-1, -1, -1) -> range(1, -1,-1) so 1,0
        dW[i] = jnp.dot(A[i].T, delta) / batch_size 
        dB[i] = jnp.sum(delta, axis=0) / batch_size


        if i > 0:
            # Backprop to previous layer:
            # delta_prev = (delta @ W[i].T) * ReLU'(Z[i-1])
            #
            # ReLU derivative (direct formula):
            # ReLU'(z) = 1 if z > 0 else 0
            #
            # Implemented as:
            # (Z[i-1] > 0).astype(float)
            #
            delta = (delta @ W[i].T) * (Z[i-1] > 0).astype(float)

    return dW, dB
    
# update
def update(W, B,dW, dB, lr):
    W_new = [w - (lr * dw) for w, dw in zip(W, dW)]
    B_new = [b - (lr * db) for b, db in zip(B, dB)]
    return W_new, B_new

# train mlp
def train_mlp(X, y, W, B, lr, epochs):
    for epoch in range(epochs):
        # X shape (4, 2)
        Z, A = forward_pass(X, W, B)
        dW, dB = backward_pass(X, y, W, B, Z, A)
        W, B = update(W, B, dW, dB, lr)
        
        if epoch % 100 == 0:
            loss = jnp.mean((A[-1] - y) ** 2)
            print(f"Epoch: {epoch} | Loss: {loss:.4f}")

    return W, B

# predict
def predict(X, W, B):
    Z, A = forward_pass(X, W, B)
    return A[-1]

# main function
if __name__ == "__main__":
    
    # XOR gate 
    X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]) # (4, 2)
    y = jnp.array([[0.0], [1.0], [1.0], [0.0]]) # (4, 1)

    # The below code will make mlp arch based on inputs. 
    # if hidden_size = 4, and hidden_layers = 1, layer_size = [2, 4, 1]
    # if hidden_size = 8, and hidden_layers = 3, layer_size = [2, 8, 8, 8, 1]
    layer_size = [2] + [hidden_size]*hidden_layers + [1]

    W, B = init_wb(layer_size, seed=42)
    W, B = train_mlp(X, y, W, B, lr=learning_rate, epochs=EPOCHS)
    predicted = predict(X, W, B)

    for i in range(len(X)):
        print(f'Input: {X[i]} | Target: {y[i]} | Prediction: {predicted[i]}')
