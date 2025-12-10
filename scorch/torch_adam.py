# torch_adam.py
from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np

from .scorch_optimizer import ScorchOptimizer, TorchParam

class TorchAdam(ScorchOptimizer):
    """
    Minimal Adam optimizer (NumPy / Scorch-friendly).

    Matches the standard Adam update:

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ** 2)

        m_hat = m_t / (1 - beta1 ** t)
        v_hat = v_t / (1 - beta2 ** t)

        param = param - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Iterable[TorchParam],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr)

        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        # One state tuple per parameter: (m, v, step)
        self._m: List[np.ndarray] = []
        self._v: List[np.ndarray] = []
        self._step: List[int] = []

        for p in self.params:
            # Lazily initialize state to zeros of matching shape
            self._m.append(np.zeros_like(p.data, dtype=np.float32))
            self._v.append(np.zeros_like(p.data, dtype=np.float32))
            self._step.append(0)

    def step(self) -> None:
        """
        Perform a single Adam update on all parameters.
        Assumes .grad has been filled for each TorchParam (or left None).
        """
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps
        lr = self.lr
        wd = self.weight_decay

        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue  # no gradient, skip

            # Optional L2 weight decay (classic, not decoupled AdamW)
            if wd != 0.0:
                g = g + wd * p.data

            m = self._m[i]
            v = self._v[i]

            # Increase step count
            self._step[i] += 1
            t = self._step[i]

            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            m[:] = beta1 * m + (1.0 - beta1) * g

            # v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ** 2)
            v[:] = beta2 * v + (1.0 - beta2) * (g * g)

            # Bias corrections
            bias_correction1 = 1.0 - (beta1 ** t)
            bias_correction2 = 1.0 - (beta2 ** t)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # Parameter update
            p.data[:] = p.data - lr * m_hat / (np.sqrt(v_hat) + eps)
