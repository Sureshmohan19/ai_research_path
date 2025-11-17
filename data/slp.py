# Single layer perceptron - understanding and training
import jax
import jax.numpy as jnp

EPOCHS = 1000

# weights and bias init
def init_wb(input_dims, seed=42):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    w = jax.random.normal(k1, (input_dims)) * 0.01 
    b = jax.random.normal(k2, ()) * 0.01
    return w, b

# loss function - just for logging
def loss_fn(w, b, X, y):
    pred = forward (X, w, b)
    return jnp.mean((pred - y) ** 2)

# activtion function
def activation(x):
    return jnp.where(x>=0, 1.0, 0.0)

# forward pass
def forward(x, w, b):
    return activation(jnp.dot(x,w) + b)

# update weights and bias
def update(w, b, X, y, lr):
    for i in range(len(X)):
        pred = forward(X[i], w, b)
        error = y[i] - pred
        w = w + (lr * error * X[i])
        b = b + (lr * error)
    return w, b
    
# training loop
def train(w, b, X, y, lr, epochs=EPOCHS):
    print(f"Initial Weigths: {w}")
    print(f"Inital Bias: {b}")
    for epoch in range(epochs):
        w, b = update(w, b, X, y, lr)
        if epoch %100 == 0:
            loss = loss_fn(w, b, X, y)
            print(f"Epoch:{epoch} | Loss:{loss:.4f} | Weights: {w} | Bias: {b:.4f}")
    print(f"Final Weights: {w}")
    print(f"Final bias: {b}")
    return w, b

# predict
def predict(w, b, X): return jnp.array([forward(x, w, b) for x in X])

# Data generation
def get_data(type = "AND"):
    if type == "AND":
        X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = jnp.array([0.0, 0.0, 0.0, 1.0])
        return X, y
    elif type == "OR":
        X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = jnp.array([0.0, 1.0, 1.0, 1.0])
        return X, y
    else: raise ValueError(f'type:{type} is not valid. Only AND or OR is possible')

# main function
if __name__ == "__main__":
    print(f"Single layer perceptron implementation...")
    print(f"_" * 40)
    
    X, y = get_data(type= "OR")
    w, b = init_wb(input_dims=2, seed=42)
    w, b = train(w, b, X, y, lr=0.1)
    prediction = predict(w, b, X)
    for i in range(len(X)):
        print(f"Input:{X[i]}, Target:{y[i]}, Prediction:{prediction[i]}")
