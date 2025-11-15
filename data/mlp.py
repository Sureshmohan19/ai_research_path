# Multi Layer Perceptron
import jax
import jax.numpy as jnp

EPOCHS = 5000

# init weights and bias
def init_wb(dim, seed=42):
    key = jax.random.PRNGKey(seed)
    W, B = [], []
    for i in range(len(dim) -1):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        w = jax.random.normal(subkey1, (dim[i], dim[i+1])) * 0.5
        b = jax.random.normal(subkey2, (dim[i+1], )) * 0.1
        W.append(w)
        B.append(b)
    return W, B

# activation function
def sigmoid(x): return 1.0/(1.0 + jnp.exp(-x))
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1.0 - sig)

def relu(x): return jnp.maximum(0, x)
def relu_derivative(x): return jnp.where(x>0, 1.0, 0.0)

# forward pass
def forward(W, B, x):
    Z = []
    A = [x]
    for i in range(len(W)):
        z = jnp.dot(A[-1], W[i]) + B[i]
        # ReLU if not output otherwise sigmoid
        a = relu(z) if i<len(W)-1 else sigmoid(z)
        Z.append(z)
        A.append(a)
    return Z, A

# backward pass
def backwards(W, B, x, y, Z, A):
    n_layers = len(W)   
    dW = [None] * n_layers
    dB = [None] * n_layers

    delta = 2 * (A[-1] - y) * sigmoid_derivative(Z[-1]) # keeping sigmoid for output

    # Backpropagate through layers
    for i in range(n_layers -1, -1, -1):
        # Gradients for current layer
        dW[i] = A[i].reshape(-1, 1) @ delta.reshape(1, -1)
        dB[i] = delta
        if i > 0:
            # ReLU for all layers except output     
            delta = (delta @  W[i].T) * relu_derivative(Z[i-1])    
    return dW, dB

# update 
def update(W, B, dW_avg, dB_avg, lr):
    W_new = [w - lr * dw for w, dw in zip(W, dW_avg)]
    B_new = [b - lr * db for b, db in zip(B, dB_avg)]
    return W_new, B_new

# loss function - only for debugging
def loss_function(W, B, X, y):
    total_loss = 0.0
    for i in range(len(X)):
        Z, A = forward(W, B, X[i])
        total_loss += jnp.mean((A[-1] - y[i]) ** 2)
    return total_loss/len(X)

# train(W, B, X, y, lr, epochs):
def train(W, B, X, y, epochs, lr): 
    n_layers = len(W)
    for epoch in range(epochs):
    
        # accumulate gradients
        dW_sum = [jnp.zeros_like(w) for w in W]
        dB_sum = [jnp.zeros_like(b) for b in B]

        for i in range(len(X)):
            Z, A = forward(W, B, X[i]) # forward pass
            dW, dB = backwards(W, B, X[i], y[i], Z, A) # backward pass
            for j in range(n_layers):
                dW_sum[j] += dW[j]
                dB_sum[j] += dB[j]

        # average gradients
        dW_avg = [dw/len(X) for dw in dW_sum]
        dB_avg = [db/len(X) for db in dB_sum]

        # update
        W, B = update(W, B, dW_avg, dB_avg, lr)

        if epoch %100 == 0:
            loss = loss_function(W, B, X, y)
            print(f"Epoch:{epoch}, Loss: {loss:.4f}")

    return W, B

# predict
def predict(W, B, X):
    pred = []
    for x in X:
        Z, A = forward(W, B, x)
        pred.append(A[-1])
    return jnp.array(pred)

# main function
if __name__ == "__main__":

    # XOR dataset
    X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = jnp.array([[0.0], [1.0], [1.0], [0.0]])

    layer_size = (2, 4, 1) # architecture
    W, B = init_wb(layer_size)
    
    print(f"MLP_architecture: {layer_size}")
    print(f"MLP_nooflayersL {len(W)}")
    
    print("\nTraining...")
    W, B = train(W, B, X, y, epochs=EPOCHS, lr=0.5) # train
    
    print("\nPredictions...")
    preds = predict(W, B, X)
    for i in range(len(X)):
        print(f"Input:{X[i]}",\
            f"Target:{y[i][0]:.1f}",\
            f"Prediction:{preds[i][0]:.4f}"
        )
