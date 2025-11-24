# Recurrent Neural Network Implementation
import jax
import jax.numpy as jnp
import wandb

raw_text = 'hello world'
hidden_size = 8
num_of_layers = 1
EPOCHS = 20
learning_rate = 0.01

# training data
def training_data(sequence):
    text = sequence # hello world
    char = sorted(list(set(text))) # [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
    vocab_size = len(char) # 8
    char_to_idx = {ch:i for i, ch in enumerate(char)} 
    # {' ': 0, 'd': 1, 'e': 2, 'h': 3, 'l': 4, 'o': 5, 'r': 6, 'w': 7}
    idx_to_char = {i:ch for i, ch in enumerate(char)}
    # {0: ' ', 1: 'd', 2: 'e', 3: 'h', 4: 'l', 5: 'o', 6: 'r', 7: 'w'}
    
    data = []
    for i in range(len(text)-1):
        input_ch = text[:i+1]
        target_ch = text[i+1]
        input_idx = [char_to_idx[ch] for ch in input_ch]
        target_idx = char_to_idx[target_ch]
        data.append((input_idx, target_idx))
    # data: 
    # [([3], 2), -> he
    #  ([3, 2], 4), -> hel 
    #  ([3, 2, 4], 4), -> hell
    #  ([3, 2, 4, 4], 5), -> hello
    #  ([3, 2, 4, 4, 5], 0), -> hello(space
    #  ([3, 2, 4, 4, 5, 0], 7), -> hello(space)w
    #  ([3, 2, 4, 4, 5, 0, 7], 5), -> hello(space)wo
    #  ([3, 2, 4, 4, 5, 0, 7, 5], 6), -> hello(space)wor
    #  ([3, 2, 4, 4, 5, 0, 7, 5, 6], 4), -> hello(space)worl
    #  ([3, 2, 4, 4, 5, 0, 7, 5, 6, 4], 1),] -> hello(space)world
    return data, vocab_size, char_to_idx, idx_to_char

# weights and bias init
def init_wb(vocab_size, hidden_size=hidden_size, num_of_layers=num_of_layers, seed=42):
    key = jax.random.PRNGKey(seed=seed) # [0 42]
    keys = jax.random.split(key, num_of_layers * 2 + 1) # 3 split
    # A JAX PRNG key is always a 2-element uint32 array:
    # [u32_low, u32_high]
    #
    # These two 32-bit integers form one 64-bit random state.
    # JAX uses this pair as the internal state for its Threefry counter-based RNG.
    # Each call to jax.random.split(key, n) returns n such keys:
    # [
    #   [a1, b1],   # key 1 (64-bit state)
    #   [a2, b2],   # key 2
    #   ...
    # ]
    # in our case, there will be three keys so, we will get something like this, 
    # [[1832780943  270669613]
    # [   64467757 2916123636]
    # [ 2465931498  255383827]]
    params = {"layers": [], "W_hy": None, "B_y": None}
    k = 0
    for layer in range(num_of_layers):
        if layer==0: input_dim = vocab_size # 8
        else: input_dim = hidden_size

        # Xavier/Glorot init
        scale_xh = jnp.sqrt(2.0 / (hidden_size + input_dim)) # 0.35355338
        scale_hh = jnp.sqrt(2.0 / (hidden_size + hidden_size)) # 0.35355338

        layer_params = {
            "W_xh": jax.random.normal(keys[k], (hidden_size, input_dim)) * scale_xh,
            "W_hh": jax.random.normal(keys[k+1], (hidden_size, hidden_size)) * scale_hh,
            "B_h": jnp.zeros(hidden_size),
            }
        params["layers"].append(layer_params)
        k += 2
    
    # Output layer: hidden state → vocabulary logits
    # Shape: (hidden_size, vocab_size)
    #
    # Why vocab_size for output dimension?
    # Character-level language model has input_dim == output_dim == vocab_size because:
    # - Input: one-hot encoded character from vocabulary (e.g., 'h' → [0,0,0,1,0,0,0,0])
    # - Output: probability distribution over same vocabulary (e.g., P('a'), P('b'), ..., P('h'))
    #
    # The model predicts the next character, so:
    # - We need one output neuron per possible character
    # - vocab_size = 8 means 8 unique characters → 8 output logits
    # - After softmax: these become probabilities for each character
    #
    # Example flow:
    # Input 'h' (vocab_size=8) → RNN → hidden state (hidden_size=4)
    #                          → W_hy @ h → logits (vocab_size=8)
    #                          → softmax → [P('a'), P('b'), ..., P('h')]
    #
    # Note: This is specific to character-level language modeling.
    scale_hy = jnp.sqrt(2.0 / (vocab_size + hidden_size)) # 0.35355338
    params["W_hy"] = jax.random.normal(keys[-1], (vocab_size, hidden_size)) * scale_hy
    params["B_y"] = jnp.zeros(vocab_size)
    params["vocab_size"] = vocab_size 
    # {
    #   'layers': [{
    #       'W_xh': (8, 8),
    #       'W_hh': (8, 8),
    #       'B_h': (8,) -> Array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    #     }],
    #   'W_hy': (8, 8),
    #   'B_y': (8,), -> Array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    #   'vocab_size': 8
    # }
    # Because input_dim = vocab_size = 8 for the first layer and hidden_size = 8, 
    # so all weight matrices map 8-dimensional vectors to 8-dimensional vectors, 
    # which gives each of W_xh, W_hh, and W_hy the shape (8, 8). 
    return params

# one-hot-encode
def one_hot_encode(index, vocab_size):
    # if index == 0, then [1. 0. 0. 0. 0. 0. 0. 0.]   # ' '
    # if index == 1, then [0. 1. 0. 0. 0. 0. 0. 0.]   # 'd'
    # if index == 2, then [0. 0. 1. 0. 0. 0. 0. 0.]   # 'e'
    # if index == 3, then [0. 0. 0. 1. 0. 0. 0. 0.]   # 'h'
    # if index == 4, then [0. 0. 0. 0. 1. 0. 0. 0.]   # 'l'
    # if index == 5, then [0. 0. 0. 0. 0. 1. 0. 0.]   # 'o'
    # if index == 6, then [0. 0. 0. 0. 0. 0. 1. 0.]   # 'r'
    # if index == 7, then [0. 0. 0. 0. 0. 0. 0. 1.]   # 'w'
    vec = jnp.zeros(vocab_size)
    vec = vec.at[index].set(1.0)
    return vec

# forward pass
def forward_pass(params, input_idx):
    num_layers = len(params["layers"]) # 1
    hidden_size = params["layers"][0]["W_hh"].shape[0] # 8
    vocab_size = params["vocab_size"] # 8
    
    h = [jnp.zeros(hidden_size) for _ in range(num_layers)]
    # h = [Array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)]
    # h: Current hidden state for each layer (gets updated at each time step)
    hidden_states = []
    # hidden_states: Stores all hidden states across all time steps (for backprop through time)
    
    # input_idx: [3] -> [3, 2] -> [3, 2, 4], ... until loop ends 
    for idx in input_idx:
        x = one_hot_encode(idx, vocab_size)
        for layer_idx in range(num_layers): # range(1) -> (0, 1) -> 0
            W_xh = params["layers"][layer_idx]["W_xh"]
            W_hh = params["layers"][layer_idx]["W_hh"]
            B_h = params["layers"][layer_idx]["B_h"]
            # Input to current hidden layer
            # - For layer_idx=0: input is the one-hot encoded character x
            # - For layer_idx>0: input is the previous layer's hidden state h[layer_idx-1]
            # This creates stacked RNN where each layer processes the output of the layer below it
            h_in = x if layer_idx == 0 else h[layer_idx - 1]
            # RNN hidden state computation
            # Formula: h_t = tanh(W_xh^T · x_t + W_hh · h_{t-1} + B_h)
            #
            # Three components combined:
            # 1. W_xh.T @ h_in: Process current input (character or previous layer output)
            # - W_xh shape: (input_dim, hidden_size), so W_xh.T is (hidden_size, input_dim)
            # - Transforms input into hidden space
            #
            # 2. W_hh @ h[layer_idx]: Process previous time step's hidden state
            # - W_hh shape: (hidden_size, hidden_size)
            # - Carries information from previous time step (memory)
            #
            # 3. B_h: Bias term
            #
            # Sum captures: current input + temporal context + bias
            h_raw = jnp.dot(W_xh, h_in) + jnp.dot(W_hh, h[layer_idx]) + B_h
            h[layer_idx] = jnp.tanh(h_raw)
        hidden_states.append(h.copy())
    
    # output layer
    top_h = h[-1]
   
    # W_xh shape: (8, 8) # input to hidden weights, takes 8-dim vector &  produces 8-dim output
    # W_hh shape: (8, 8) # hidden to hidden weights, maps 8-dim hidden st to 8-dim hidden st
    # B_h shape: (8,)    # bias for the hidden layer, one value per hidden unit
    # h_in shape: (8,)   # the input going into the layer, always an 8-dim vector
    #                    # (either one-hot input or previous layer's hidden state)
    # jnp.dot(W_xh.T, h_in) → (8,)    # multiplying (8×8) by (8,) gives an 8-dim vector
    # jnp.dot(W_hh, h[layer_idx])     → (8,)
    # h_raw shape: (8,)               # sum of two 8-dim vectors + bias
    # h[layer_idx] shape: (8,)        # final hidden state of this layer
    # top_h = h[-1]
    # top_h shape: (8,)               # hidden state of the last layer, still 8-dim
    # hidden_states: list of hidden states across time
    # each entry is a copy of all layer hidden vectors → shapes inside: (num_layers, 8)

    return top_h, hidden_states
        
# softmax
def softmax(logits):
    # exp_logits shape: (8,)         # exponentiated logits, still 8 numbers
    # softmax output shape: (8,)     # probability for each of the 8 vocab items
    exp_logits = jnp.exp(logits - jnp.max(logits))
    return exp_logits / jnp.sum(exp_logits)

# loss function
def loss_fn(params, h_final, target_idx):
    # W_hy shape: (8, 8)             # maps hidden state (8,) to output logits (8,)
    # B_y shape: (8,)                # output bias, one value per vocab item
    W_hy = params["W_hy"]
    B_y = params["B_y"]

    # h_final shape: (8,)            # final hidden state from RNN
    # jnp.dot(h_final, W_hy) → (8,)  # produces 8 logits, one per vocab symbol
    # logits shape: (8,)             # raw scores before softmax
    logits = jnp.dot(W_hy, h_final) + B_y
    
    # probs shape: (8,)              # probability distribution over 8 vocabulary symbols
    probs = softmax(logits)
    
    # Cross-Entropy Loss (CEL)
    # Formula: L = -log(p_target)
    # Where p_target is the predicted probability of the correct class
    #
    # For one-hot encoded target [0,0,1,0,0,0,0,0] with target_idx=2:
    # - We only care about probs[2] (probability assigned to correct class)
    # - Loss = -log(probs[target_idx])
    # - Lower probability → higher loss (good predictions get low loss)
    # - 1e-8 added for numerical stability (avoid log(0))
    #
    # This is Cross-Entropy Loss for single sample classification
    # loss is a single number: shape ()
    loss = -jnp.log(probs[target_idx] + 1e-8)
    return loss, probs

# backward pass
def backward_pass(params, input_idx, target_idx, probs, hidden_states):
    num_layers = len(params["layers"]) # 1
    hidden_size = params["layers"][0]["W_hh"].shape[0] # 8
    vocab_size = params["vocab_size"] # 8

    layer_grads = []
    # layer_grads list creation:
    # dW_xh shape: (8, 8)            # same shape as W_xh
    # dW_hh shape: (8, 8)            # same shape as W_hh
    # dB_h shape: (8,)               # same shape as B_h

    # output layer grads:
    # dW_hy shape: (8, 8)            # same shape as W_hy
    # dB_y shape: (8,)               # same shape as B_y
    for l in range(num_layers):
        layer_grads.append({
            "dW_xh": jnp.zeros_like(params["layers"][l]["W_xh"]),
            "dW_hh": jnp.zeros_like(params["layers"][l]["W_hh"]),
            "dB_h": jnp.zeros_like(params["layers"][l]["B_h"]),
            })
    dW_hy = jnp.zeros_like(params["W_hy"])
    dB_y = jnp.zeros_like(params["B_y"])
   
    # d_logits shape: (8,)           # gradient of logits for each vocab item
    # h_final shape: (8,)            # final hidden state from last layer at last time step
    # jnp.outer(h_final, d_logits) → dW_hy shape: (8, 8)
    # dB_y = d_logits
    # dB_y shape: (8,)               # gradient for each vocab item
    #
    # Derivation of ∂L/∂logits for Softmax + Cross-Entropy Loss
    #
    # Step 1: Define the functions
    # Softmax: p_i = exp(z_i) / Σ_j exp(z_j)  where z = logits
    # Cross-Entropy Loss: L = -log(p_target) = -log(p_k) where k is target_idx
    #
    # Step 2: Apply chain rule
    # ∂L/∂z_i = ∂L/∂p_j · ∂p_j/∂z_i  (sum over all j)
    #
    # Step 3: Compute ∂L/∂p_j (loss derivative w.r.t. probabilities)
    # L = -log(p_k)
    # ∂L/∂p_j = -1/p_k  if j = k (target class)
    #         = 0       if j ≠ k (non-target classes)
    #
    # Step 4: Compute ∂p_j/∂z_i (softmax derivative)
    # Case 1: If j = i (derivative of p_i w.r.t. its own logit z_i):
    #   ∂p_i/∂z_i = p_i(1 - p_i)
    #
    # Case 2: If j ≠ i (derivative of p_j w.r.t. different logit z_i):
    #   ∂p_j/∂z_i = -p_i · p_j
    #
    # Step 5: Combine using chain rule
    # ∂L/∂z_i = Σ_j (∂L/∂p_j · ∂p_j/∂z_i)
    #
    # When i = k (target class logit):
    # ∂L/∂z_k = (∂L/∂p_k) · (∂p_k/∂z_k)
    #         = (-1/p_k) · p_k(1 - p_k)
    #         = -(1 - p_k)
    #         = p_k - 1
    #
    # When i ≠ k (non-target class logit):
    # ∂L/∂z_i = (∂L/∂p_k) · (∂p_k/∂z_i)
    #         = (-1/p_k) · (-p_i · p_k)
    #         = p_i
    #
    # Step 6: Final result
    # ∂L/∂z_i = p_i - 1  if i = target_idx
    #         = p_i      if i ≠ target_idx
    #
    # In vector form: ∂L/∂logits = probs - target_one_hot
    #
    # Example: target_idx=2, probs=[0.1, 0.2, 0.5, 0.2]
    # target_one_hot = [0, 0, 1, 0]
    # gradient = [0.1, 0.2, 0.5, 0.2] - [0, 0, 1, 0] = [0.1, 0.2, -0.5, 0.2]
    d_logits = probs.copy()
    d_logits = d_logits.at[target_idx].add(-1.0)
    h_final = hidden_states[-1][-1]
   
    # Why outer product for dW_hy?
    #
    # Forward pass dimensions:
    # h_final: (hidden_size,)    e.g., (8,)
    # W_hy:    (hidden_size, vocab_size)    e.g., (8, )# logits = h_final @ W_hy + B_y
    # logits:  (vocab_size,)     e.g., (8,)
    #
    # We need: ∂L/∂W_hy with shape (hidden_size, vocab_size) to match W_hy
    #
    # Chain rule: ∂L/∂W_hy = ∂L/∂logits · ∂logits/∂W_hy
    #
    # The forward operation logits = h @ W_hy means:
    # logits[j] = Σ_i h[i] * W_hy[i,j]
    #
    # Therefore: ∂logits[j]/∂W_hy[i,j] = h[i]
    #
    # So: ∂L/∂W_hy[i,j] = ∂L/∂logits[j] · h[i]
    #                    = h[i] · d_logits[j]
    #
    # This is exactly the outer product: h ⊗ d_logits
    dW_hy = jnp.outer(d_logits, h_final)
    
    # Why just d_logits for dB_y?
    #
    # Forward pass: logits = h @ W_hy + B_y
    #
    # Chain rule: ∂L/∂B_y = ∂L/∂logits · ∂logits/∂B_y
    #
    # Since bias is simply added: ∂logits/∂B_y = 1 (identity)
    #
    # Therefore: ∂L/∂B_y = d_logits · 1 = d_logits
    #
    # Shape naturally matches: d_logits is (vocab_size,), same as B_y
    dB_y = d_logits
    
    # grads["layers"] → list of 1 layer dict, each with shapes (8,8), (8,8), (8,)
    # grads["W_hy"] shape: (8, 8)
    # grads["B_y"] shape: (8,)
    grads = {"layers": [], "W_hy": dW_hy, "B_y": dB_y}
    
    # params: ["W_hy"] shape: (8, 8) | d_logits shape: (8,)
    # dh_next shape: (8,)          # gradient flowing into final hidden state
    #
    # Backpropagate gradient from logits to final hidden state
    # Chain rule: ∂L/∂h = ∂L/∂logits · ∂logits/∂h
    #
    # Forward pass: logits = h @ W_hy + B_y
    # Therefore: ∂logits/∂h = W_hy
    #
    dh_next = jnp.dot(params["W_hy"], d_logits)

    for t in range(len(input_idx)-1, -1, -1):
        for l in reversed(range(num_layers)):
            # h_current: shape: (8,) | hidden state at time t, layer l
            # h_prev shape: (8,)     | previous hidden state (or zeros at t=0)
            h_current = hidden_states[t][l]
            h_prev = hidden_states[t-1][l] if t>0 else jnp.zeros(hidden_size)

            # dh_next shape: (8,)
            # h_current shape: (8,)
            # dh_raw shape: (8,)           # gradient after tanh derivative
            
            # Gradient through tanh activation
            # Chain rule: ∂L/∂h_raw = ∂L/∂h · ∂h/∂h_raw
            #
            # Forward pass: h = tanh(h_raw)
            # Where: h_raw = W_xh.T @ x + W_hh @ h_prev + B_h
            #
            # Derivative of tanh:
            # tanh(z) = (e^z - e^-z) / (e^z + e^-z)
            # d(tanh(z))/dz = 1 - tanh²(z)
            #               = 1 - h²
            #
            # Therefore:
            # ∂h/∂h_raw = 1 - tanh²(h_raw) = 1 - h_current²
            #
            # Chain rule:
            # dh_raw = dh_next · (1 - h_current²)
            #
            # dh_next: gradient flowing to this hidden state (from layer above or time step ahead)
            # (1 - h_current²): tanh derivative, determines how much gradient passes through
            # dh_raw: gradient w.r.t. pre-activation (before tanh was applied)
            #
            # Shape: (8,) - element-wise multiplication
            dh_raw = dh_next * (1 - h_current**2)

            # Gradient w.r.t. bias B_h
            # Forward: h_raw = W_xh.T @ x + W_hh @ h_prev + B_h
            # Derivative: ∂h_raw/∂B_h = 1 (bias is just added)
            # Chain rule: ∂L/∂B_h = ∂L/∂h_raw · ∂h_raw/∂B_h = dh_raw · 1 = dh_raw
            # Shape: (8,) matches B_h shape
            layer_grads[l]["dB_h"] += dh_raw
            
            # Gradient w.r.t. recurrent weights W_hh
            # Forward: h_raw = W_xh.T @ x + W_hh @ h_prev + B_h
            # The W_hh @ h_prev term contributes to h_raw
            # Derivative: ∂h_raw/∂W_hh = h_prev (same logic as dW_hy earlier)
            # Chain rule: ∂L/∂W_hh[i,j] = ∂L/∂h_raw[j] · h_prev[i]
            #                            = dh_raw[j] · h_prev[i]
            # This is outer product: outer(h_prev, dh_raw)
            # Shape: (8, 8) matches W_hh shape
            layer_grads[l]["dW_hh"] += jnp.outer(dh_raw, h_prev)

            # x shape: (8,)
            # Get input to this layer at time t
            # - First layer (l=0): input is one-hot encoded character
            # - Deeper layers (l>0): input is hidden state from layer below
            x = one_hot_encode(input_idx[t], vocab_size) if l==0 else hidden_states[t][l-1]

            # x shape: (8,)
            # dh_raw shape: (8,)
            # outer(x, dh_raw) shape: (8, 8)
            # dW_xh shape: (8, 8)
            
            # Gradient w.r.t. input weights W_xh
            # Forward: h_raw = W_xh.T @ x + W_hh @ h_prev + B_h
            # The W_xh.T @ x term contributes to h_raw
            # Derivative: ∂h_raw/∂W_xh = x (input that was multiplied with these weights)
            # Chain rule: ∂L/∂W_xh = outer(x, dh_raw)
            # Shape: (8, 8) matches W_xh shape
            layer_grads[l]["dW_xh"] += jnp.outer(dh_raw, x)

            # W_hh shape: (8, 8)
            # dh_raw shape: (8,)
            # dh_next shape: (8,)          # gradient passed to earlier timestep
            
            # Backpropagate gradient through time
            # Forward: h_raw(t) = ... + W_hh @ h_prev(t-1) + ...
            # We need gradient w.r.t. h_prev to continue backprop to t-1
            # Chain rule: ∂L/∂h_prev = ∂L/∂h_raw · ∂h_raw/∂h_prev
            #                        = dh_raw · W_hh
            # Shape: W_hh is (8,8), dh_raw is (8,), result is (8,)
            # This dh_next will be used for the previous time step (t-1)
            dh_next = jnp.dot(params["layers"][l]["W_hh"], dh_raw)

    # grads["layers"][l]["dW_xh"] shape: (8, 8)
    # grads["layers"][l]["dW_hh"] shape: (8, 8)
    # grads["layers"][l]["dB_h"] shape: (8,)
    for l in range(num_layers):
        grads["layers"].append({
            "dW_xh": layer_grads[l]["dW_xh"],
            "dW_hh": layer_grads[l]["dW_hh"],
            "dB_h": layer_grads[l]["dB_h"],
    })

    # grads = {
    #     "layers": [{                    # list length = num_layers (here 1)
    #             "dW_xh": (8, 8),        # gradient of W_xh
    #             "dW_hh": (8, 8),        # gradient of W_hh
    #             "dB_h":  (8,)           # gradient of B_h
    #         }],
    #     "W_hy": (8, 8),                 # gradient of output weight matrix
    #     "B_y": (8,)                     # gradient of output bias
    #   }
    return grads

# update
def update(params, grads, lr=0.1):
    # ORIGINAL PARAM SHAPES:
    # W_xh: (8, 8)
    # W_hh: (8, 8)
    # B_h:  (8,)
    # W_hy: (8, 8)
    # B_y:  (8,)

    for l in range(len(params["layers"])):
        params["layers"][l]["W_xh"] -= lr * grads["layers"][l]["dW_xh"]
        params["layers"][l]["W_hh"] -= lr * grads["layers"][l]["dW_hh"]
        params["layers"][l]["B_h"] -= lr * grads["layers"][l]["dB_h"]
    params["W_hy"] -= lr * grads["W_hy"]
    params["B_y"] -= lr * grads["B_y"]
    
    # UPDATE OPERATIONS:
    # (8, 8) -= (8, 8)    # W_xh update
    # (8, 8) -= (8, 8)    # W_hh update
    # (8,)   -= (8,)      # B_h update
    # (8, 8) -= (8, 8)    # W_hy update
    # (8,)   -= (8,)      # B_y update
    return params

# train
def train_rnn(data, params, hidden_size, vocab_size, epoch, lr=learning_rate):
    # wandb init
    wandb.init(
        project="rnn_research",
        config={
            "hidden_size": {hidden_size},
            "vocab_size": {vocab_size},
            "num_layers": {num_of_layers},
            "learning_rate": {learning_rate},
            "text_length": {len(raw_text)},
            "epoch": {EPOCHS},
        })
    for epo in range(epoch):
        total_loss = 0
        for input_idx, target_idx in data:
            h_final, hidden_states = forward_pass(params, input_idx)
            loss, probs = loss_fn(params, h_final, target_idx)
            total_loss += loss
            grads = backward_pass(params, input_idx, target_idx, probs, hidden_states)
            params = update(params, grads, lr)
        
        # Gradient norms
        grad_norms = {
            "grad_W_xh_norm": float(jnp.linalg.norm(grads["layers"][0]["dW_xh"])),
            "grad_W_hh_norm": float(jnp.linalg.norm(grads["layers"][0]["dW_hh"])),
            "grad_W_hy_norm": float(jnp.linalg.norm(grads["W_hy"])),
        }
        
        # Weight norms
        weight_norms = {
            "weight_W_xh_norm": float(jnp.linalg.norm(params["layers"][0]["W_xh"])),
            "weight_W_hh_norm": float(jnp.linalg.norm(params["layers"][0]["W_hh"])),
            "weight_W_hy_norm": float(jnp.linalg.norm(params["W_hy"])),
        }
        
        # Hidden state stats
        h_final = hidden_states[-1][0]
        hidden_stats = {
            "h_mean": float(jnp.mean(h_final)),
            "h_std": float(jnp.std(h_final)),
            "h_max": float(jnp.max(jnp.abs(h_final))),
        }
        
        # Update sizes
        update_sizes = {
            "update_W_hh_norm": float(jnp.linalg.norm(lr * grads["layers"][0]["dW_hh"])),
        }
        
        # Log everything
        wandb.log({
            "loss": float(total_loss),
            "epoch": epo,
            **grad_norms,
            **weight_norms,
            **hidden_stats,
            **update_sizes,
        })
        
        # Weight histograms every 50 epochs
        if epo % 50 == 0:
            wandb.log({
                "W_hh_hist": wandb.Histogram(params["layers"][0]["W_hh"]),
                "W_xh_hist": wandb.Histogram(params["layers"][0]["W_xh"]),
            })

        if epo % 10 == 0: print(f"Epoch:{epo}, Loss:{total_loss:.4f}")
    return params
    
# generate
def generate(params, start, ctoi, itoc, num_chars=10):
    num_layers = len(params["layers"]) # 1
    hidden_size = params["layers"][0]["W_hh"].shape[0] # 8
    vocab_size = params["vocab_size"] # 8

    generated_ = start
    
    # h list shape: [ (8,) ]        # one hidden state per layer
    # x shape: (8,)                 # one-hot
    h = [jnp.zeros(hidden_size) for _ in range(num_layers)]
    x = one_hot_encode(ctoi[start], vocab_size)

    # FIRST FORWARD PASS (initial char)
    # W_xh: (8, 8)
    # W_hh: (8, 8)
    # B_h:  (8,)
    # h_in: (8,)
    # jnp.dot(W_xh.T, h_in): (8,)
    # jnp.dot(W_hh, h[l]):   (8,)
    # h_raw: (8,)
    # h[l]:   (8,)
    for l in range(num_layers):
        W_xh = params["layers"][l]["W_xh"]
        W_hh = params["layers"][l]["W_hh"]
        B_h = params["layers"][l]["B_h"]
        h_in = x if l == 0 else h[l-1]
        h_raw = jnp.dot(W_xh, h_in) + jnp.dot(W_hh, h[l]) + B_h
        h[l] = jnp.tanh(h_raw)

    # GENERATION LOOP
    # top_h: (8,)
    # logits: (8,)
    # probs:  (8,)
    for _ in range(num_chars - 1):
        top_h = h[-1]
        logits = jnp.dot(params["W_hy"], top_h) + params["B_y"]
        probs = softmax(logits)

        next_idx = int(jnp.argmax(probs))
        next_ch = itoc[next_idx]
        generated_ += next_ch

        # for next round
        # x (next one-hot): (8,)
        # h_in: (8,)
        # h_raw: (8,)
        # h[l]: (8,)
        x = one_hot_encode(next_idx, vocab_size)
        for l in range(num_layers):
            W_xh = params["layers"][l]["W_xh"]
            W_hh = params["layers"][l]["W_hh"]
            B_h = params["layers"][l]["B_h"]
            h_in = x if l==0 else h[l-1]
            h_raw = jnp.dot(W_xh, h_in) + jnp.dot(W_hh, h[l]) + B_h
            h[l] = jnp.tanh(h_raw)

    return generated_

# main function
if __name__ == "__main__":
    data, vocab_size, char_to_idx, idx_to_char = training_data(sequence=raw_text)
    params = init_wb(vocab_size, hidden_size, num_of_layers, seed=42)
    params = train_rnn(data, params, hidden_size, vocab_size, epoch=EPOCHS)
    output = generate(params, 'h', char_to_idx, idx_to_char, num_chars=10)
    print(output)
    wandb.finish()
