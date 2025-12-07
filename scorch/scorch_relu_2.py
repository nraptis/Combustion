# scorch/scorch_relu_2.py
from __future__ import annotations
import numpy as np

from scorch.scorch_module import ScorchModule


class ScorchReLU2(ScorchModule):
    """
    Elementwise ReLU activation:
        y = max(0, x)

    Forward:
        - Saves a mask of where x > 0
    Backward:
        - Passes gradients only where x was positive
        - Zeros gradients where x <= 0

    No parameters, no parameter gradients.
    """

    def __init__(self, name: str | None = None):
        super().__init__()
        self.name = name or "ScorchReLU2"
        self._mask: np.ndarray | None = None

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, x):
        """
        x: np.ndarray, any shape
        returns: np.ndarray, same shape
        """
        x_arr = np.asarray(x, dtype=np.float32)

        # 1 where x > 0, 0 where x <= 0
        self._mask = (x_arr > 0).astype(np.float32)

        # Apply ReLU elementwise
        return x_arr * self._mask

    # ------------------------------------------------------
    # Backward
    # ------------------------------------------------------
    def backward(self, grad_output):
        """
        grad_output:
            dL/dy from the next layer.
        Returns:
            dL/dx, same shape.
        """
        if self._mask is None:
            raise RuntimeError(f"{self.name}: backward() called before forward().")

        g_out = np.asarray(grad_output, dtype=np.float32)

        if g_out.shape != self._mask.shape:
            raise ValueError(
                f"{self.name}: grad_output shape {g_out.shape} "
                f"does not match cached mask shape {self._mask.shape}"
            )

        # Chain rule:
        # dL/dx = dL/dy * (x > 0 ? 1 : 0)
        return g_out * self._mask

    # ------------------------------------------------------
    # Params
    # ------------------------------------------------------
    def parameters(self):
        return []

    def zero_grad(self):
        # No params → no parameter grads
        pass

    def __repr__(self):
        return f"{self.name}()"
