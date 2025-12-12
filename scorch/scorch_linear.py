# scorch/scorch_linear.py
from __future__ import annotations
import math
import numpy as np

from scorch.scorch_functional import linear_forward
from scorch.scorch_module import ScorchModule


class ScorchLinear(ScorchModule):
    """
    A minimal fully-connected layer:
        y = W @ x + b

    Shapes (conceptually):
        Input features:  D = in_features
        Output features: C = out_features

    Forward:
        x: (..., D)
        W: (C, D)
        b: (C,)
        y: (..., C)

        That is:
            - if x is 1-D, shape (D,), y is (C,)
            - if x is 2-D, shape (N, D), y is (N, C)
            - if x is 3-D, shape (B, T, D), y is (B, T, C)
            - etc.

    Backward:
        grad_output: dL/dy, same shape as y = (..., C)
        Produces:
            grad_W: dL/dW, shape (C, D)
            grad_b: dL/db, shape (C,)
            grad_input: dL/dx, shape (..., D)
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

        # Cache for backward: we need x with its full shape
        self._last_x: np.ndarray | None = None

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------
    def forward(self, x):
        """
        x: np.ndarray of shape (..., in_features)
        returns: np.ndarray of shape (..., out_features)
        """
        x_arr = np.asarray(x, dtype=np.float32)

        if x_arr.ndim < 1:
            raise ValueError(
                f"{self.name}: Expected at least 1-D input (..., D), got {x_arr.shape}"
            )

        if x_arr.shape[-1] != self.in_features:
            raise ValueError(
                f"{self.name}: Expected last dimension {self.in_features}, "
                f"got {x_arr.shape[-1]}"
            )

        # Save for backward (full shape, not flattened)
        self._last_x = x_arr.copy()

        # ------------------------------------------------------------------
        # Conceptual NumPy one-liner (kept for reference)
        #
        #   out = x_arr @ self.W.T + self.b
        #
        # However, we expand this into manual Python loops so each multiply /
        # add is explicit for learning and debugging.
        # ------------------------------------------------------------------

        # Handle the simple 1-D case (D,) directly via linear_forward:
        if x_arr.ndim == 1:
            # (Old behavior, still here and still readable)
            # return self.W @ x_arr + self.b
            return linear_forward(x_arr, self.W, self.b)

        # General N-D case: (..., D)
        original_shape = x_arr.shape          # (..., D)
        leading_shape = original_shape[:-1]   # ...
        D = self.in_features
        C = self.out_features

        # Flatten all leading dimensions into a single "batch" dimension.
        batch_size = int(np.prod(leading_shape)) if leading_shape else 1

        x_flat = x_arr.reshape(batch_size, D)            # (B, D)
        out_flat = np.zeros((batch_size, C), dtype=np.float32)  # (B, C)

        # Manual forward: for each sample n, and each output neuron i,
        # compute dot(W[i, :], x[n, :]) + b[i]
        for n in range(batch_size):          # each sample in the batch
            for i in range(C):              # each output neuron
                acc = 0.0
                for j in range(D):          # each input feature
                    acc += self.W[i, j] * x_flat[n, j]
                acc += self.b[i]
                out_flat[n, i] = acc

        # Reshape back to (..., C)
        out = out_flat.reshape(*leading_shape, C)
        return out

    # ------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------
    def backward(self, grad_output):
        """
        grad_output: dL/dy, same shape as forward output (..., out_features)

        Computes:
            grad_W (accumulated into self.grad_W)   shape (C, D)
            grad_b (accumulated into self.grad_b)   shape (C,)
        Returns:
            grad_input: dL/dx, shape (..., in_features)
        """
        if self._last_x is None:
            raise RuntimeError(f"{self.name}: backward called before forward.")

        g_out = np.asarray(grad_output, dtype=np.float32)
        x = self._last_x

        if g_out.ndim < 1:
            raise ValueError(
                f"{self.name}: grad_output must be at least 1-D (..., C), "
                f"got {g_out.shape}"
            )

        if g_out.shape[-1] != self.out_features:
            raise ValueError(
                f"{self.name}: grad_output last dimension {g_out.shape[-1]} "
                f"does not match (out_features,) = ({self.out_features},)"
            )

        if x.shape[:-1] != g_out.shape[:-1]:
            raise ValueError(
                f"{self.name}: grad_output leading shape {g_out.shape[:-1]} "
                f"does not match input leading shape {x.shape[:-1]}"
            )

        D = self.in_features
        C = self.out_features

        # ------------------------------------------------------------------
        # 1-D special case: keep it as close as possible to your old code
        # ------------------------------------------------------------------
        if x.ndim == 1 and g_out.ndim == 1:
            # x: (D,)
            # g_out: (C,)

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

        # ------------------------------------------------------------------
        # General N-D case: x shape (..., D), grad_output shape (..., C)
        #
        # We flatten to:
        #   x_flat:       (B, D)
        #   g_out_flat:   (B, C)
        #
        # Then:
        #   grad_W[i,j] = Σ_n  g_out[n,i] * x[n,j]
        #   grad_b[i]   = Σ_n  g_out[n,i]
        #   grad_input[n,j] = Σ_i W[i,j] * g_out[n,i]
        # ------------------------------------------------------------------
        leading_shape = x.shape[:-1]
        batch_size = int(np.prod(leading_shape)) if leading_shape else 1

        x_flat = x.reshape(batch_size, D)            # (B, D)
        g_out_flat = g_out.reshape(batch_size, C)    # (B, C)

        grad_W = np.zeros((C, D), dtype=np.float32)
        grad_b = np.zeros((C,), dtype=np.float32)
        grad_input_flat = np.zeros((batch_size, D), dtype=np.float32)

        # Manual accumulation over the batch and feature dimensions
        for n in range(batch_size):       # each sample
            for i in range(C):           # each output neuron
                gi = g_out_flat[n, i]    # gradient flowing into neuron i
                grad_b[i] += gi          # dL/db accumulates over batch

                # grad_W[i,j] = Σ_n g[n,i] * x[n,j]
                # grad_input[n,j] = Σ_i W[i,j] * g[n,i]
                for j in range(D):
                    xnj = x_flat[n, j]
                    grad_W[i, j] += gi * xnj
                    grad_input_flat[n, j] += self.W[i, j] * gi

        # Reshape grad_input back to (..., D)
        grad_input = grad_input_flat.reshape(*leading_shape, D)

        # Accumulate gradients (so multiple samples / batches can add up)
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
