# scorch/scorch_linear.py
from __future__ import annotations
import math
import numpy as np

from scorch.nn_functional import linear_forward
from scorch.scorch_module import ScorchModule


class ScorchLinear(ScorchModule):
    """
    A minimal fully-connected layer:
        y = W @ x + b

    This is a proper ScorchModule:
        - forward() defined
        - backward() will be added later
        - parameters() returns [W, b]
        - zero_grad() resets gradient buffers
    """

    def __init__(self, in_features: int, out_features: int, name: str | None = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.name = name or f"ScorchLinear({in_features}->{out_features})"

        # --------------------------------------------------
        # Parameter initialization (Xavier-like)
        # --------------------------------------------------
        limit = 1.0 / math.sqrt(self.in_features)
        self.W = np.random.uniform(
            -limit, +limit,
            size=(self.out_features, self.in_features)
        ).astype(np.float32)

        self.b = np.zeros((self.out_features,), dtype=np.float32)

        # Gradients (filled during backward pass)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Cache for backward (save input x)
        self._last_x: np.ndarray | None = None

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------
    def forward(self, x):
        """
        x: np.ndarray of shape (in_features,)
        returns: np.ndarray of shape (out_features,)
        """
        x_arr = np.asarray(x, dtype=np.float32)

        if x_arr.ndim != 1:
            raise ValueError(f"{self.name}: Expected input shape (D,), got {x_arr.shape}")

        if x_arr.shape[0] != self.in_features:
            raise ValueError(
                f"{self.name}: Expected {self.in_features} features, got {x_arr.shape[0]}"
            )

        # Save for backward pass
        self._last_x = x_arr.copy()

        return linear_forward(x_arr, self.W, self.b)

    # ------------------------------------------------------
    # Backward pass (stub for now)
    # ------------------------------------------------------
    def backward(self, grad_output):
        """
        grad_output: np.ndarray of shape (out_features,)
        Should compute:
            grad_W
            grad_b
            grad_input
        But we'll implement this later.
        """
        raise NotImplementedError("ScorchLinear.backward is not implemented yet.")

    # ------------------------------------------------------
    # Parameter + gradient handling
    # ------------------------------------------------------
    def parameters(self):
        return [self.W, self.b]

    def zero_grad(self):
        self.grad_W.fill(0.0)
        self.grad_b.fill(0.0)

    def __repr__(self):
        return f"{self.name}(W={self.W.shape}, b={self.b.shape})"
