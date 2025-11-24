# Gated Recurrent Unit implementation for sin wave prediction (toy example)
import jax
import jax.numpy as jnp

hidden_dim = 8
input_dim = 1
output_dim = 1
T = 200
EPOCHS = 500
learning_rate = 0.001

# init_params
def init_params(key, in_dim, out_dim, hid_dim):
    # example keys structure
    # [[1797259609 2579123966]
    # [ 928981903 3453687069]
    # [4146024105 2718843009]
    # [2467461003 3840466878]
    # [2285895361  433833334]
    # [1524306142 1887795613]
    # [3792494674 2909014575]
    # [2716826189  292468403]]
    keys = jax.random.split(key, 8)

    params = {
        # Reset gate
        # W_r : (hidden_dim, input_dim) = (8, 1)
        # U_r : (hidden_dim, hidden_dim) = (8, 8)
        # b_r : (hidden_dim,) = (8,)
        "W_r" : jax.random.normal(keys[0], (hid_dim, in_dim)) * jnp.sqrt(2.0 / (hid_dim + in_dim)),
        "U_r" : jax.random.normal(keys[1], (hid_dim, hid_dim)) * jnp.sqrt(2.0 / (hid_dim + hid_dim)),
        "b_r" : jnp.zeros((hid_dim,)),

        # Update gate
        # W_z : (hidden_dim, input_dim) = (8, 1)
        # U_z : (hidden_dim, hidden_dim) = (8, 8)
        # b_z : (hidden_dim,) = (8,)
        "W_z" : jax.random.normal(keys[2], (hid_dim, in_dim)) * jnp.sqrt(2.0 / (hid_dim + in_dim)),
        "U_z" : jax.random.normal(keys[3], (hid_dim, hid_dim)) * jnp.sqrt(2.0 / (hid_dim + hid_dim)),
        "b_z" : jnp.zeros((hid_dim,)),
       
        # Candidate
        # W_h : (hidden_dim, input_dim) = (8, 1)
        # U_h : (hidden_dim, hidden_dim) = (8, 8)
        # b_h : (hidden_dim,) = (8,)
        "W_h" : jax.random.normal(keys[4], (hid_dim, in_dim)) * jnp.sqrt(2.0 / (hid_dim + in_dim)),
        "U_h" : jax.random.normal(keys[5], (hid_dim, hid_dim)) * jnp.sqrt(2.0 / (hid_dim + hid_dim)),
        "b_h" : jnp.zeros((hid_dim,)),
        
        # Ouput
        # W_y : (output_dim, hidden_dim) = (1, 8)
        # b_y : (output_dim,) = (1,)
        "W_y" : jax.random.normal(keys[6], (out_dim, hid_dim)) * jnp.sqrt(2.0 / (out_dim + hid_dim)),
        "b_y" : jnp.zeros((out_dim,))
        }
    return params

# sin wave sequence
def sin_seq(T):
    # t          : (T,)        # time steps
    # x          : (T,)        # sin values
    # input_seq  : (T-1, 1)    # x[0..T-2], reshaped to column vectors
    # target_seq : (T-1, 1)    # x[1..T-1], reshaped to column vectors
    t = jnp.linspace(0, 20, T)
    x = jnp.sin(t)
    return x[:-1].reshape(-1, 1), x[1:].reshape(-1, 1)

# activation functions
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)
    
# dtanh
def dtanh(x):
    t = jnp.tanh(x)
    return 1.0 - t * t

# forward pass
def forward_gru(params, x_seq):
    T = x_seq.shape[0] # 199
    hidden_dim = params["b_r"].shape[0] # 8
    
    # h: hidden state - the "memory" of the RNN
    # Initialize to zeros for the first time step (t=0)
    # This means the network starts with no prior context/memory
    # As we process the sequence, h gets updated at each time step
    # carrying information forward through time
    h = jnp.zeros((hidden_dim, )) # (8, )
    
    preds = []
    cache = {
        "x_t"       : [],       # Input at each timestep
        "h_prev"    : [],       # Hidden state before update
        "h_t"       : [],       # Hidden state after update
        "r_t"       : [],       # Reset gate activation
        "z_t"       : [],       # Update gate activation
        "candidate" : [],       # Candidate hidden state
        "s_r"       : [],       # Reset gate pre-activation
        "s_z"       : [],       # Update gate pre-activation
        "s_h"       : [],       # Candidate pre-activation
        "y_t"       : []        # Output at each timestep
    }

    for t in range(T): # (0, 199)
        x_t = x_seq[t]

        # Gate pre-activations
        # General form: output = weights × input + bias
        #
        # GRU gates need TWO inputs combined:
        # 1. Current input x_t
        # 2. Previous hidden state h
        #
        # Derivation for Reset Gate:
        # We want: f(x_t, h) → (8,) output
        #
        # Step 1: Transform x_t to hidden space
        # W_r @ x_t  where W_r: (8, 1), x_t: (1,) → result: (8,)
        #
        # Step 2: Transform h (already in hidden space)
        # U_r @ h    where U_r: (8, 8), h: (8,) → result: (8,)
        #
        # Step 3: Combine both transformations + bias
        # s_r = W_r @ x_t + U_r @ h + b_r
        #
        # This is: output = W₁×input₁ + W₂×input₂ + bias
        #          (generalized wx + b for multiple inputs)
        s_r = params["W_r"] @ x_t + params["U_r"] @ h + params["b_r"]
        
        # Same derivation for Update Gate (different weights, same structure)
        s_z = params["W_z"] @ x_t + params["U_z"] @ h + params["b_z"]

        # Gates activation
        # Apply sigmoid to get gate values in [0, 1]
        r_t = sigmoid(s_r)
        z_t = sigmoid(s_z)

        # Candidate hidden state (proposal for new memory)
        # 
        # Step 1: Apply reset gate to old memory
        # - r_t * h: Filters old hidden state based on relevance
        # - If r_t ≈ 1: Keep past memory (pattern continues)
        # - If r_t ≈ 0: Forget past memory (fresh start)
        #
        # Step 2: Combine filtered memory with current input
        # s_h = W_h @ x_t + U_h @ (r_t * h) + b_h
        # - W_h @ x_t: Process current input
        # - U_h @ (r_t * h): Process relevant past memory
        # - This creates a "proposal" for what new memory should be
        #
        # Step 3: Apply tanh activation
        # - Squashes to [-1, 1] range
        # - candidate: Proposed new hidden state based on current context
        s_h = params["W_h"] @ x_t + params["U_h"] @ (r_t * h) + params["b_h"]
        candidate = jnp.tanh(s_h)

        # Final hidden state (blend old and new)
        #
        # Update gate z_t controls the mixing ratio:
        # - z_t * h: Keep this much of OLD memory
        # - (1 - z_t) * candidate: Accept this much of NEW candidate
        #
        # Each dimension can blend differently:
        # - If z_t[i] ≈ 1: Dimension i keeps old memory (ignore new info)
        # - If z_t[i] ≈ 0: Dimension i updates heavily (accept new info)
        # - If z_t[i] ≈ 0.5: Dimension i blends 50-50
        #
        # This weighted average lets GRU selectively update its memory
        h_new = (z_t * h) + ((1 - z_t) * candidate)

        # Output layer
        # Transform final hidden state to prediction
        # y_t: Predicted next value in sequence (e.g., next sin wave value)
        y_t = params["W_y"] @ h_new + params["b_y"]
        preds.append(y_t)

        # Store everything in cache for BPTT
        cache["x_t"].append(x_t)
        cache["s_r"].append(s_r) 
        cache["h_prev"].append(h)
        cache["h_t"].append(h_new)
        cache["s_z"].append(s_z)
        cache["candidate"].append(candidate)
        cache["r_t"].append(r_t)
        cache["z_t"].append(z_t)
        cache["s_h"].append(s_h)
        cache["y_t"].append(y_t)

        # Update hidden state for next time step
        # h becomes h_new, carrying forward the updated memory
        # This h will be used as "previous hidden state" when processing the next input
        h = h_new
    return jnp.stack(preds, axis=0), cache

# backward pass
def backward_gru(params, preds, cache, y_seq):
    grads = {k: jnp.zeros_like(v) for k, v in params.items()}
    T = preds.shape[0]

    # MSE loss
    # loss = 0.5 * jnp.sum((preds - targets) ** 2) but we can calculate its derivative directly
    # dL_dpred = preds - target
    dL_dpred = preds - y_seq

    # Initial loss of hidden state wrt loss
    dL_dh_next = jnp.zeros_like(cache["h_t"][0])

    # BPTT
    for t in reversed(range(T)):
        x_t      = cache["x_t"][t]
        h_prev   = cache["h_prev"][t]
        h_t      = cache["h_t"][t]
        r_t      = cache["r_t"][t]
        z_t      = cache["z_t"][t]
        candidate = cache["candidate"][t]
        s_r      = cache["s_r"][t]
        s_z      = cache["s_z"][t]
        s_h      = cache["s_h"][t]
        
        # Outer layer gradients
        dL_dy = dL_dpred[t]
        grads["W_y"] += jnp.outer(dL_dy, h_t)
        grads["b_y"] += dL_dy
        dL_dh = params["W_y"].T @ dL_dy + dL_dh_next  
        
        # GRU hidden gradients
        dL_dz = dL_dh * (h_prev - candidate)
        dL_candidate = dL_dh * (1 - z_t)
        dL_dh_prev = dL_dh * z_t

        # Candidate path
        dL_ds_h = dL_candidate * dtanh(s_h)
        grads["W_h"] += jnp.outer(dL_ds_h, x_t)
        grads["U_h"] += jnp.outer(dL_ds_h, r_t * h_prev)
        grads["b_h"] += dL_ds_h

        # Contribution to h_prev
        dL_dh_prev += (params["U_h"].T @ dL_ds_h) * r_t

        # Contribution to r_t from candidate
        dL_dr = (params["U_h"].T @ dL_ds_h) * h_prev

        # Reset gate: r_t = sigmoid(s_r)
        dL_ds_r = dL_dr * dsigmoid(s_r)
        grads["W_r"] += jnp.outer(dL_ds_r, x_t)
        grads["U_r"] += jnp.outer(dL_ds_r, h_prev)
        grads["b_r"] += dL_ds_r

        dL_dh_prev += params["U_r"].T @ dL_ds_r

        # Update gate: z_t = sigmoid(s_z)
        dL_ds_z = dL_dz * dsigmoid(s_z)
        grads["W_z"] += jnp.outer(dL_ds_z, x_t)
        grads["U_z"] += jnp.outer(dL_ds_z, h_prev)
        grads["b_z"] += dL_ds_z

        dL_dh_prev += params["U_z"].T @ dL_ds_z

        # Propagate to next iteration
        dL_dh_next = dL_dh_prev
    return grads

# update
def update_gru(params, grads, lr):
    return {k: params[k] - lr * grads[k] for k in params}

# train
def train_gru(params, x_seq, y_seq, lr, epochs):
    for epoch in range(epochs):
        preds, cache = forward_gru(params, x_seq)
        grads = backward_gru(params, preds, cache, y_seq)
        params = update_gru(params, grads, lr)

        loss = jnp.mean(0.5 * (preds - y_seq) ** 2)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.4f}")
    return params
    
# main function
if __name__ == "__main__":
    key = jax.random.PRNGKey(seed=0) # [0 0]
    params = init_params(key, input_dim, output_dim, hidden_dim)
    input_seq, target_seq = sin_seq(T) #(199, 1) (199, 1)
    params = train_gru(params, input_seq, target_seq, lr=learning_rate, epochs=EPOCHS)
