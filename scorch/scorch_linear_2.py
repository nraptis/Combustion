# scorch/scorch_linear_2.py
from __future__ import annotations

import math
import numpy as np

from scorch.scorch_module import ScorchModule


class ScorchLinear2(ScorchModule):
    """
    Fully-connected (dense) layer:

        y = W @ x + b

    where:
        x: (D,)           input features
        W: (C, D)         weight matrix
        b: (C,)           bias vector
        y: (C,)           output activations

    Backprop equations (single sample):

        Given:
            g_out = dL/dy   shape (C,)

        Gradients:
            dL/dW = g_out[:, None] @ x[None, :]     (outer product)
            dL/db = g_out
            dL/dx = W^T @ g_out
    """

    def __init__(self, in_features: int, out_features: int, name: str | None = None):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.name = name or f"ScorchLinear2({in_features}->{out_features})"

        # Xavier-ish uniform init:
        # W_ij ~ U(-1/sqrt(D), 1/sqrt(D))
        limit = 1.0 / math.sqrt(self.in_features)

        self.W = np.random.uniform(
            -limit, +limit,
            size=(self.out_features, self.in_features),
        ).astype(np.float32)

        self.b = np.zeros((self.out_features,), dtype=np.float32)

        # Gradient buffers
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Cache for backward
        self._last_x: np.ndarray | None = None

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, x):
        """
        x: np.ndarray of shape (in_features,)
        returns: np.ndarray of shape (out_features,)
        """
        x_arr = np.asarray(x, dtype=np.float32)

        if x_arr.ndim != 1:
            raise ValueError(
                f"{self.name}: Expected input of shape (D,), got {x_arr.shape}"
            )

        if x_arr.shape[0] != self.in_features:
            raise ValueError(
                f"{self.name}: Expected {self.in_features} features, "
                f"got {x_arr.shape[0]}"
            )

        # Cache input for backprop
        self._last_x = x_arr.copy()

        # y = W @ x + b
        # W: (C, D), x: (D,) -> y: (C,)
        y = self.W @ x_arr + self.b

        return y

    # ------------------------------------------------------
    # Backward
    # ------------------------------------------------------
    def backward(self, grad_output):
        """
        grad_output:
            Gradient of loss with respect to the output y.
            Shape: (out_features,)

        Computes:
            grad_W: accumulated into self.grad_W
            grad_b: accumulated into self.grad_b

        Returns:
            grad_input: gradient of loss with respect to input x, shape (in_features,)
        """
        if self._last_x is None:
            raise RuntimeError(f"{self.name}: backward() called before forward().")

        g_out = np.asarray(grad_output, dtype=np.float32)

        if g_out.shape != (self.out_features,):
            raise ValueError(
                f"{self.name}: grad_output shape {g_out.shape} "
                f"does not match (out_features,) = ({self.out_features},)"
            )

        x = self._last_x  # (D,)

        # --- Gradients by "well-known" linear layer math ---

        # 1) dL/dW = outer(g_out, x)
        #    Each weight W[i,j] sees:
        #      dL/dW[i,j] = g_out[i] * x[j]
        grad_W = np.outer(g_out, x)   # (C, D)

        # 2) dL/db = g_out
        #    Because y[i] = ... + b[i], so ∂y[i]/∂b[i] = 1
        grad_b = g_out                # (C,)

        # 3) dL/dx = W^T @ g_out
        #    Each input x[j] collects contributions from every output neuron i:
        #      dL/dx[j] = Σ_i g_out[i] * W[i,j]
        grad_input = self.W.T @ g_out  # (D,)

        # Accumulate into buffers so multiple samples can add up
        self.grad_W += grad_W
        self.grad_b += grad_b

        return grad_input

    # ------------------------------------------------------
    # Parameter / grad helpers
    # ------------------------------------------------------
    def parameters(self):
        """
        Return a flat list of learnable arrays.
        """
        return [self.W, self.b]

    def zero_grad(self):
        """
        Reset gradient buffers to zero.
        """
        self.grad_W.fill(0.0)
        self.grad_b.fill(0.0)

    def __repr__(self):
        return f"{self.name}(W={self.W.shape}, b={self.b.shape})"
