# scorch/grad_check.py
from __future__ import annotations

import numpy as np

from scorch.scorch_sequential import ScorchSequential
from scorch.scorch_linear_2 import ScorchLinear2
from scorch.scorch_relu_2 import ScorchReLU2


def softmax_and_cross_entropy_with_grad(logits: np.ndarray, target_index: int):
    """
    Same as in your runner, but local here for convenience.
    """
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)

    loss = -np.log(probs[target_index] + 1e-12)

    grad = probs.copy()
    grad[target_index] -= 1.0

    return float(loss), grad


def compute_loss(model: ScorchSequential, x: np.ndarray, target_index: int) -> float:
    """
    Forward through the model and return scalar loss for a single (x, y).
    No gradient computation here, pure forward.
    """
    logits = model.forward(x)  # (C,)
    loss, _ = softmax_and_cross_entropy_with_grad(logits, target_index)
    return loss


def compute_loss_and_backprop(model: ScorchSequential, x: np.ndarray, target_index: int):
    """
    Forward + backward for a single sample.
    Returns:
        loss (float)
    and fills model's gradients via backward().
    """
    model.zero_grad()

    logits = model.forward(x)
    loss, grad_logits = softmax_and_cross_entropy_with_grad(logits, target_index)
    _ = model.backward(grad_logits)

    return loss


def grad_check_weight(
    model: ScorchSequential,
    x: np.ndarray,
    target_index: int,
    layer_idx: int,
    i: int,
    j: int,
    eps: float = 1e-4,
):
    """
    Compare analytic grad_W[i,j] vs numeric finite-difference gradient on a given layer.

    layer_idx:
        index in model.layers where the ScorchLinear2 lives.

    i, j:
        indices into that layer's W matrix.
    """
    layer = model.layers[layer_idx]
    if not isinstance(layer, ScorchLinear2):
        raise TypeError(f"Layer at index {layer_idx} is not ScorchLinear2: {type(layer)}")

    W = layer.W

    # --- 1) Analytic gradient via backprop ---
    _ = compute_loss_and_backprop(model, x, target_index)
    analytic = layer.grad_W[i, j]

    # --- 2) Numeric gradient via finite differences ---
    original = W[i, j]

    # W[i, j] + eps
    W[i, j] = original + eps
    loss_plus = compute_loss(model, x, target_index)

    # W[i, j] - eps
    W[i, j] = original - eps
    loss_minus = compute_loss(model, x, target_index)

    # Restore original weight
    W[i, j] = original

    numeric = (loss_plus - loss_minus) / (2.0 * eps)

    # --- 3) Print comparison ---
    print(f"[grad_check] layer={layer_idx}, W[{i},{j}]")
    print(f"  analytic: {analytic}")
    print(f"  numeric : {numeric}")
    print(f"  diff    : {abs(analytic - numeric)}")

    return analytic, numeric
