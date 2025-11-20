# Single Layer Perceptron Implementation - Vectorized way
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
    
    # return (4, )
    return jnp.where(x>= 0.0, 1.0, 0.0)

# forward pass
def forward_pass(X, w, b):
    
    sum_ = jnp.dot(X, w) + b
    # jnp.dot(X, w) ->
    # jnp.dot ((4, 2), (2, )) -> shape (4, )
    # (4, ) + () -> (4, )
    return activation(sum_)

# train
def train_slp(X, y, w, b, lr, epochs):
    
    # naive for loop method 
    for epoch in range(epochs): # range(500) -> (0, 500)
        #    
        # X -> (4, 2) 
        # w -> [ 0.00075926 -0.00486343] with shape (2, )
        # b -> 0.00605764 with shape ()
        #
        y_pred = forward_pass(X, w, b) # (4, )

        # MSE Loss function for a single sample:
        # L = 1/2 * (y - pred)^2
        # 
        # ∂L/∂pred = (y - pred) * (-1)
        # ∂L/∂pred = - (y - pred)
        # ∂L/∂pred = pred - y
        #
        # we name dL/dpred as error_der here
        error_der  = y_pred - y # (4, )

        # Derivative wrt weights w
        # ∂L/∂w = (∂L/∂pred) * (∂pred/∂w)
        # 
        # Step 1: ∂L/∂pred
        # ∂L/∂pred = pred - y # found above
        #
        # Step 2: ∂pred/∂w
        # pred = w * X  + b
        # ∂pred/∂w = X
        #
        # Final:
        # ∂L/∂w = (pred - y) * X
        #
        # SGD update for weights
        # w <- w - lr * ∂L/∂w
        #
        grad_w = X.T @ error_der # X.T (2, 4) @ (4, ) -> shape (2,)
        w = w - lr * grad_w 

        # Derivative wrt bias b
        # ∂L/∂b = (∂L/∂pred) * (∂pred/∂b)
        # 
        # Step 1: ∂L/∂pred  (same as above)
        # ∂L/∂pred = pred - y
        #
        # Step 2: ∂pred/∂b
        # pred = w · X + b
        # ∂pred/∂b = 1
        #
        # Final:
        # ∂L/∂b = (pred - y) * 1
        # ∂L/∂b = pred - y
        # 
        # SGD update for bias
        # b <- b - lr * ∂L/∂b
        b = b - (lr * jnp.mean(error_der)) # (), ( () * () ) = ()

    
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
    predicted = forward_pass(X, w, b)
    
    for i in range(len(X)):
        print(f"VInput: {X[i]} | Target: {y[i]} | Predicted: {predicted[i]}")

