# torch_optimizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import numpy as np


@dataclass
class TorchParam:
    """
    Minimal 'parameter' wrapper for Scorch-style tensors.

    data: np.ndarray holding the actual weights.
    grad: np.ndarray of the same shape, filled by your backward pass.
    """
    data: np.ndarray
    grad: np.ndarray | None = None


class TorchOptimizer:
    """
    Minimal PyTorch-style optimizer base class.

    - Holds a flat list of TorchParam objects (or anything with .data / .grad).
    - Provides zero_grad().
    - Child classes implement step().
    """

    def __init__(self, params: Iterable[TorchParam], lr: float = 1e-3) -> None:
        self.params: List[TorchParam] = list(params)
        if len(self.params) == 0:
            raise ValueError("TorchOptimizer got an empty parameter list.")
        self.lr: float = float(lr)

    def zero_grad(self) -> None:
        """
        Set all gradients to zero (in-place), if they exist.
        """
        for p in self.params:
            if p.grad is not None:
                p.grad[...] = 0.0

    def step(self) -> None:
        """
        Perform a single optimization step.

        Must be overridden by subclasses.
        """
        raise NotImplementedError("TorchOptimizer.step() must be implemented by subclasses.")
