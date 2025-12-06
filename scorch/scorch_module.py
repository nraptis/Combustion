# scorch/scorch_module.py
from __future__ import annotations

from typing import List, Any


class ScorchModule:
    """
    Minimal base class for all Scorch layers.

    Design goals:
      - Very small surface area.
      - Forward and backward are explicit.
      - Parameter handling is unified (so optimizers can just call .parameters()).

    Subclasses should override:
      - forward(self, x)
      - backward(self, grad_output)
      - parameters(self)         (if they have learnable params)
      - zero_grad(self)          (if they store gradients)
    """

    def forward(self, x: Any) -> Any:
        """
        Compute the forward pass.

        Must be overridden in subclasses.

        Example signature in subclasses:
            x: np.ndarray
            returns: np.ndarray

        Forward = compute values
        Backward = compute sensitivities
        """
        raise NotImplementedError(f"{self.__class__.__name__}.forward not implemented.")

    def backward(self, grad_output: Any) -> Any:
        """
        Compute the backward pass.

        grad_output is dL/d(out) from the next layer.
        This method must compute dL/d(input) and store
        gradients for parameters (if any).

        Chain Rule:
        (dL / dx) = (dL / dy) * (dy / dx)

        (The gradient leaving a node) =
        (The gradient entering that node) *
        (the derivative of the operation the node performs)

        Forward = compute values
        Backward = compute sensitivities
        """
        raise NotImplementedError(f"{self.__class__.__name__}.backward not implemented.")

    # --------------------------------------------------
    # Parameter handling
    # --------------------------------------------------
    def parameters(self) -> List[Any]:
        """
        Return a flat list of all learnable parameters
        owned by this module.

        Layers without parameters can just inherit this
        default (empty list).
        """
        return []

    def zero_grad(self) -> None:
        """
        Reset gradients for all learnable parameters.

        Layers without parameters (e.g. ReLU, MaxPool)
        can inherit this default no-op implementation.
        Layers with parameters should override and zero
        their internal grad arrays.
        """
        # No-op by default
        pass

    # --------------------------------------------------
    # Convenience: allow calling module(x) like in PyTorch
    # --------------------------------------------------
    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
