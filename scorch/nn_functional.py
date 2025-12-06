# scorch/nn_functional.py
from __future__ import annotations
import numpy as np

def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """

    W is the matrix of learned weights.
    linear_forward applies those weights to
    the input features to compute the output
    neuron activations. These activations
    propagate forward to the next layer,
    or become the final “logits” used
    by the network to make predictions.

    Pure function: y = W @ x + b

    x: (D,)
    W: (C, D)
    b: (C,)
    returns: (C,)
    """

    # return W @ x + b

    C = W.shape[0]   # number of output neurons
    D = W.shape[1]   # number of input features

    # Safety check
    if x.shape[0] != D:
        raise ValueError(f"linear_forward: Expected x of shape ({D},), got {x.shape}")

    z = [0.0 for _ in range(C)]

    for i in range(C):         # for every output neuron
        acc = 0.0
        for j in range(D):     # multiply across the row of W
            acc += W[i, j] * x[j]
        acc += b[i]            # add bias
        z[i] = acc

    return np.array(z, dtype=np.float32)

def softmax_cross_entropy(logits: np.ndarray, target_index: int) -> float:
    """
    Pure function: softmax + cross-entropy loss for a single example.

    logits is the output from linear forward … a 1-D list of floats.
    target_index is the correct class label as an integer.

    logits: (C,)
    target_index: int in [0, C)

    Classes:
        0 = cat
        1 = dog
        2 = fish

    logits = [1.5, 0.2, -0.3]
    target_index = 2 (this is expected to be "fish")

    Interpretation:
        cat: 1.5
        dog: 0.2
        fish: -0.3

    ...

    exp([1.5, 0.2, -0.3]) → normalized → probs

    probs ≈ [0.65, 0.23, 0.12]

    Which means:
        Model thinks “cat” with probability 65%
        Model thinks “dog” with probability 23%
        Model thinks “fish” with probability 12%

    This was expected as fish, so it will be a high loss...

    loss = - log(prob_of_correct_class)
    loss = - log(0.12)
    loss ≈ 2.12  (a high loss)

    """
    # For numerical stability
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)

    # Cross entropy loss: -log p_target
    loss = -np.log(probs[target_index] + 1e-12)
    return float(loss)
