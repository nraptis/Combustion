# scorch/scorch_relu.py
from __future__ import annotations

import numpy as np
from scorch.scorch_module import ScorchModule


class ScorchReLU(ScorchModule):
    """
    Elementwise ReLU activation:

        y = max(0, x)

    Forward:
        - Works on any np.ndarray shape.
        - Stores a mask (x > 0) for backward.

    Backward:
        - grad_input = grad_output * (x > 0)
    """

    def __init__(self, name: str | None = None):
        super().__init__()
        self.name = name or "ScorchReLU"
        self._mask: np.ndarray | None = None

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------
    def forward(self, x):
        """
        x: np.ndarray (any shape)
        returns: np.ndarray (same shape)
        """
        x_arr = np.asarray(x, dtype=np.float32)

        # Store mask for backward: 1 where x > 0, 0 elsewhere
        self._mask = (x_arr > 0).astype(np.float32)

        # ReLU: max(0, x) == x * (x > 0)
        return x_arr * self._mask

    # ------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------
    def backward(self, grad_output):
        """
        grad_output: dL/dy, same shape as forward output
        returns: dL/dx, same shape as input

        For ReLU:
            dL/dx = dL/dy * 1 where x > 0
            dL/dx = dL/dy * 0 where x <= 0
        """
        if self._mask is None:
            raise RuntimeError(f"{self.name}: backward called before forward.")

        grad_out_arr = np.asarray(grad_output, dtype=np.float32)

        if grad_out_arr.shape != self._mask.shape:
            raise ValueError(
                f"{self.name}: grad_output shape {grad_out_arr.shape} "
                f"does not match mask shape {self._mask.shape}"
            )

        # Elementwise multiply with mask
        grad_input = grad_out_arr * self._mask
        return grad_input

    # ------------------------------------------------------
    # Parameters / grads
    # ------------------------------------------------------
    def parameters(self):
        # ReLU has no learnable parameters
        return []

    def zero_grad(self):
        # Nothing to do; no parameters
        pass

    def __repr__(self):
        return f"{self.name}()"
