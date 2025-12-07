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

    Shapes:
        x: (D,)
        W: (C, D)
        b: (C,)
        y: (C,)

    Backward:
        grad_output: dL/dy, shape (C,)
        Produces:
            grad_W: dL/dW, shape (C, D)
            grad_b: dL/db, shape (C,)
            grad_input: dL/dx, shape (D,)
    """

    def __init__(self, in_features: int, out_features: int, name: str | None = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.name = name or f"ScorchLinear({in_features}->{out_features})"

        # Parameter initialization (Xavier-ish)
        limit = 1.0 / math.sqrt(self.in_features)
        self.W = np.random.uniform(
            -limit, +limit,
            size=(self.out_features, self.in_features)
        ).astype(np.float32)

        self.b = np.zeros((self.out_features,), dtype=np.float32)

        # Gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Cache for backward
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

        # Save for backward
        self._last_x = x_arr.copy()

        return linear_forward(x_arr, self.W, self.b)

    # ------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------
    def backward(self, grad_output):
        """
        grad_output: dL/dy, shape (out_features,)

        Computes:
            grad_W (accumulated into self.grad_W)
            grad_b (accumulated into self.grad_b)
        Returns:
            grad_input: dL/dx, shape (in_features,)
        """
        if self._last_x is None:
            raise RuntimeError(f"{self.name}: backward called before forward.")

        g_out = np.asarray(grad_output, dtype=np.float32)

        if g_out.shape != (self.out_features,):
            raise ValueError(
                f"{self.name}: grad_output shape {g_out.shape} "
                f"does not match (out_features,) = ({self.out_features},)"
            )

        x = self._last_x  # shape (D,)

        D = self.in_features
        C = self.out_features

        # dL/dW = outer(grad_output, x)
        # grad_W = np.outer(g_out, x)  # (C, D)

        # grad_W is the rate of change in error loss with respect to weight.
        # ...
        # The derivative of the loss with respect to weight W[i,j].
        # ...
        # grad_W[i,j] ==> “How much would the total loss change if we
        # nudged weight W[i,j] upward by a tiny amount?”

        grad_W = np.zeros((C, D), dtype=np.float32)

        for i in range(C):        # for each output neuron
            for j in range(D):    # for each input feature
                grad_W[i][j] = g_out[i] * x[j]

        # dL/db = grad_output
        grad_b = g_out  # (C,)

        # dL/dx = W^T @ grad_output
        # ...
        # grad_input = Wᵀ @ g_out because each input
        # feature receives error signals from every output
        # neuron, scaled by the weight that connects them...
        # and that weighted sum is a dot product.

        # grad_input = self.W.T @ g_out  # (D,)

        grad_input = np.zeros(D, dtype=np.float32)

        for j in range(D):               # for each input feature x[j]
            acc = 0.0
            for i in range(C):           # sum over all output neurons
                acc += self.W[i][j] * g_out[i]
            grad_input[j] = acc

        # Accumulate gradients (so multiple samples can add up)
        self.grad_W += grad_W
        self.grad_b += grad_b

        return grad_input

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
