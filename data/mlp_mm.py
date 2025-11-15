# Multi Layer Perceptron using Matrix Multiplication
import jax
import jax.numpy as jnp

N = 1
EPOCHS = 5000

# init wb
def init_wb(dims, seed=42):
    key = jax.random.PRNGKey(seed) 
    W, B = [], []
    for i in range(len(dims)-1):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        w = jax.random.normal(subkey1, (dims[i], dims[i+1])) * 0.5
        b = jax.random.normal(subkey2, (dims[i+1])) * 0.1
        W.append(w)
        B.append(b)
    return W, B

# activation function
def sigmoid(x): return 1.0 / (1.0 + jnp.exp(-x))
def relu(x): return jnp.maximum(0.0, x)

# forward_batch
def forward_batch(W, B, X_batch):
    Z = []
    A = [X_batch]
    for i in range(len(W)):
        z = jnp.dot(A[-1], W[i]) + B[i]
        a = relu(z) if i<len(W)-1 else sigmoid(z)
        Z.append(z)
        A.append(a)
    return Z, A

# backward_batch
def backward_batch(W, B, X_batch, y_batch, Z, A):
    n_layers = len(W)
    batch_size = X_batch.shape[0]
    dW = [None] * n_layers
    dB = [None] * n_layers

    delta = 2 * (A[-1] - y_batch) * (A[-1] * (1-A[-1]))

    for i in range(n_layers-1, -1, -1):
        dW[i] = jnp.dot(A[i].T, delta) / batch_size
        dB[i] = jnp.sum(delta, axis=0) / batch_size

        if i > 0:
            delta = jnp.dot(delta, W[i].T) * (A[i]>0).astype(jnp.float32)
    return dW, dB

# update weights
def update_weights(W, B, dW, dB, lr):
    W_new = [w-(lr*dw) for w, dw in zip(W, dW)]
    B_new = [b-(lr*db) for b, db in zip(B, dB)]
    return W_new, B_new
    
# training 
def train(W, B, X, y, epoch, lr):
    for i in range(epoch):
        Z, A = forward_batch(W, B, X)
        dW, dB = backward_batch(W, B, X, y, Z, A)
        W, B = update_weights(W, B, dW, dB, lr)
        if i % 100 == 0:
            loss = jnp.mean((A[-1]-y) ** 2)
            print(f"Epoch:{i}, Loss:{loss:.4f}")
    return W, B

# predict
def predict(W, B, X_batch):
    Z, A = forward_batch(W, B, X_batch)
    return A[-1]

# main function
if __name__ == "__main__":
    # XOR data
    X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = jnp.array([[0.0], [1.0], [1.0], [0.0]])
    
    layer_size = [2]+[4]*N+[1] # 2input, 4hidden, 1output
    W, B = init_wb(layer_size, seed=42)
    W, B = train(W, B, X, y, epoch=EPOCHS, lr=0.5)    
    preds = predict(W, B, X)
    for i in range(len(X)):
        print(f"Input: {X[i]} " \
            f"Target: {y[i][0]:.1f} " \
            f"Predcition: {preds[i][0]:.4f} " \
            )
