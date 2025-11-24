# Single Layer Perceptron Implementation
import jax
import jax.numpy as jnp

learning_rate = 0.01  # ()
EPOCHS = 1000 # ()

# init weights and bias
def init_wb(X, y):
    key = jax.random.PRNGKey(seed=42) 
    # key -> [0 42] 

    k1, k2 = jax.random.split(key, 2)
    # k1 -> [1832780943  270669613] 
    # k2 -> [  64467757 2916123636]

    scaling_factor = 0.01

    w = jax.random.normal(k1, X[1].shape) * scaling_factor
    # w = [ 0.00075926 -0.00486343]
    # w.shape -> (2, )
    
    b = jax.random.normal(k2, ()) * scaling_factor
    # b -> 0.00605764
    # b.shape -> ()

    return w, b

# activation
def activation(x):
    
    # return scalar value ()
    return jnp.where(x>= 0.0, 1.0, 0.0)

# forward pass
def forward_pass(x, w, b):
    
    sum_ = (jnp.dot(x, w) + b)
    # jnp.dot(x, w) ->
    # jnp.dot ((2, ), (2, )) -> shape (), a scalar 
    # () + () -> ()
    return activation(sum_)

# train
def train_slp(X, y, w, b, lr, epochs):
    
    # naive for loop method 
    for epoch in range(epochs): # range(500) -> (0, 500)
        for i in range(len(X)): # len(X) = 4
            # for first loop with above written w and b random values
            #    
            # X[i] -> X[0] -> [0., 0.] with shape (2, ) 
            # w = [ 0.00075926 -0.00486343] with shape (2, )
            # b = 0.00605764 with shape ()
            # 
            y_pred = forward_pass(X[i], w, b) # scalar value ()

            # MSE Loss function for a single sample:
            # L = 1/2 * (y[i] - pred)^2
            # 
            # ∂L/∂pred = (y[i] - pred) * (-1)
            # ∂L/∂pred = - (y[i] - pred)
            # ∂L/∂pred = pred - y[i]
            #
            # we name dL/dpred as error_der here
            error_der  = y_pred - y[i] # scalar value ()

            # Derivative wrt weights w
            # ∂L/∂w = (∂L/∂pred) * (∂pred/∂w)
            # 
            # Step 1: ∂L/∂pred
            # ∂L/∂pred = pred - y[i] # found above
            #
            # Step 2: ∂pred/∂w
            # pred = Σ_j  w_j * x[i]_j  + b
            # ∂pred/∂w = x[i]
            #
            # Final:
            # ∂L/∂w = (pred - y[i]) * x[i]
            #
            # SGD update for weights
            # w <- w - lr * ∂L/∂w
            w = w - (lr * error_der * X[i]) # () - ( () * () * () ) = ()

            # Derivative wrt bias b
            # ∂L/∂b = (∂L/∂pred) * (∂pred/∂b)
            # 
            # Step 1: ∂L/∂pred  (same as above)
            # ∂L/∂pred = pred - y[i]
            #
            # Step 2: ∂pred/∂b
            # pred = w · x[i] + b
            # ∂pred/∂b = 1
            #
            # Final:
            # ∂L/∂b = (pred - y[i]) * 1
            # ∂L/∂b = pred - y[i]
            # 
            # SGD update for bias
            # b <- b - lr * ∂L/∂b
            b = b - (lr * error_der) # (), ( () * () ) = ()
    
        if epoch % 100 == 0:
            loss = forward_pass(X, w, b)
            # X -> (4, 2)
            # w -> (2, )
            # b -> ()
            #
            # forward pass is just jnp.dot(X, w) + b
            # jnp.dot((4, 2), (2, )) + () = (4, ) which matches with y (4, )

            mean_loss = jnp.mean((loss - y) ** 2)
            # jnp.mean((4, ) - (4, ) ** 2) = jnp.mean((4,) ** 2) = () produces scalar value

            print(f"Epoch: {epoch} | Loss: {mean_loss:.4f}")
            
    return w, b

# main function
if __name__ == "__main__":
    
    X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]) 
    # X.shape -> (4, 2)

    y = jnp.array([0.0, 0.0, 0.0, 1.0])
    # y.shape -> (4, )

    w, b = init_wb(X, y)
    # w.shape -> (2, )
    # b.shape -> ()

    # train
    w, b = train_slp(X, y, w, b, lr=learning_rate, epochs=EPOCHS)
    # w.shape -> (2, )
    # b.shape -> ()

    # predict
    # X has 4 samples, so the loop runs 4 times.
    # Each forward_pass(x, w, b) returns one number.
    # Collecting them gives: [pred1, pred2, pred3, pred4]
    predicted = jnp.array([forward_pass(x, w, b) for x in X])
    
    for i in range(len(X)):
        print(f"Input: {X[i]} | Target: {y[i]} | Predicted: {predicted[i]}")

