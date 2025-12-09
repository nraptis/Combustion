# scorch/scorch_loss.py
from __future__ import annotations

from typing import Optional, Union
import numpy as np

from .scorch_module import ScorchModule
from .scorch_reduction import ScorchReduction

# Allow both "mean"/"sum"/"none" and ScorchReduction.MEAN etc.
ReductionLike = Union[str, ScorchReduction]


class ScorchLoss(ScorchModule):
    """
    Base class for Scorch loss functions.

    - Subclasses should implement forward(input, target).
    - Training code should NOT call .backward() on loss modules in this framework.
      Instead, use dedicated functional helpers (e.g. cross_entropy_with_grad)
      that return both (loss, grad_input).

    Attributes:
        reduction: ScorchReduction enum (NONE | MEAN | SUM)
    """

    def __init__(self, reduction: ReductionLike = ScorchReduction.MEAN) -> None:
        super().__init__()

        # Normalize to enum
        if isinstance(reduction, ScorchReduction):
            self.reduction: ScorchReduction = reduction
        else:
            # assume string-like
            try:
                self.reduction = ScorchReduction(str(reduction).lower())
            except Exception as e:
                raise ValueError(
                    f"Invalid reduction: {reduction!r}. "
                    f"Expected one of: {[r.value for r in ScorchReduction]}"
                ) from e

    # PyTorch-style call semantics: loss(input, target)
    def forward(self, input: np.ndarray, target: np.ndarray):
        """
        Compute the loss.

        Subclasses must override this method.

        Typical behavior:
          - If reduction == NONE: return per-sample losses, shape (N,).
          - If reduction == MEAN or SUM: return a scalar float.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.forward not implemented.")

    def backward(self, grad_output) -> None:
        """
        For now we DO NOT support calling backward() on loss modules.

        In Scorch, the typical pattern is:
            loss, grad_logits = cross_entropy_with_grad(logits, targets)

        So if this gets called accidentally, we want a loud failure,
        not a silent no-op.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.backward is not supported. "
            "Use explicit functional helpers that return (loss, grad_input)."
        )

    # --------------------------------------------------
    # Helpers for subclasses
    # --------------------------------------------------
    def _reduce(self, per_sample: np.ndarray) -> float | np.ndarray:
        """
        Apply reduction (NONE | MEAN | SUM) to a 1D array of per-sample losses.
        """
        per_sample = np.asarray(per_sample, dtype=np.float32)

        # If already scalar, just return float
        if per_sample.ndim == 0:
            return float(per_sample)

        match self.reduction:
            case ScorchReduction.MEAN:
                if per_sample.size == 0:
                    # You could choose np.nan here to mimic PyTorch more strictly
                    return 0.0
                return float(per_sample.mean())

            case ScorchReduction.SUM:
                if per_sample.size == 0:
                    return 0.0
                return float(per_sample.sum())

            case ScorchReduction.NONE:
                return per_sample

            case _:
                # Should be impossible if __init__ is correct
                raise RuntimeError(f"Unknown reduction: {self.reduction}")

    def parameters(self):
        """
        Loss modules don't have learnable parameters in this engine.
        """
        return []

    def zero_grad(self) -> None:
        """
        Nothing to reset for stateless loss modules.
        """
        # No-op; keeps ScorchModule contract satisfied.
        pass


class ScorchWeightedLoss(ScorchLoss):
    """
    Base class for losses that support per-class weights, like CrossEntropyLoss.

    Attributes:
        weight: Optional np.ndarray of shape (num_classes,)
    """

    def __init__(
        self,
        weight: Optional[np.ndarray] = None,
        reduction: ReductionLike = ScorchReduction.MEAN,
    ) -> None:
        super().__init__(reduction=reduction)

        if weight is not None:
            w = np.asarray(weight, dtype=np.float32)
            if w.ndim != 1:
                raise ValueError(
                    f"weight must be 1D (num_classes,), got shape {w.shape}"
                )
            self.weight: Optional[np.ndarray] = w
        else:
            self.weight = None


# -------------------------------------------------------------------
# Functional: cross_entropy_with_grad
# -------------------------------------------------------------------

def cross_entropy_with_grad(
    logits: np.ndarray,
    targets: np.ndarray,
    weight: Optional[np.ndarray] = None,
    reduction: ReductionLike = ScorchReduction.MEAN,
) -> tuple[float | np.ndarray, np.ndarray]:
    """
    Cross-entropy loss + gradient w.r.t. logits (like torch.nn.CrossEntropyLoss).

    Args:
        logits:  (N, C) float32 - raw scores from the final layer.
        targets: (N,) int64     - class indices in [0, C).
        weight:  Optional (C,) float32 - per-class weights.
        reduction: "none" | "mean" | "sum" or ScorchReduction enum.

    Returns:
        loss:
            - scalar float if reduction is "mean" or "sum"
            - (N,) float32 array if reduction is "none"
        grad_logits: (N, C) float32 - dL/d(logits)
    """
    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)

    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape (N, C), got {logits.shape}")
    if targets.ndim != 1:
        raise ValueError(f"Expected targets shape (N,), got {targets.shape}")

    N, C = logits.shape
    if targets.shape[0] != N:
        raise ValueError(
            f"Targets length {targets.shape[0]} does not match batch size {N}"
        )

    # Normalize reduction to enum
    if isinstance(reduction, ScorchReduction):
        reduction_enum = reduction
    else:
        reduction_enum = ScorchReduction(str(reduction).lower())

    # ------------------------------------
    # Softmax probabilities (stable)
    # ------------------------------------
    shifted = logits - np.max(logits, axis=1, keepdims=True)  # (N, C)
    exp = np.exp(shifted, dtype=np.float32)
    probs = exp / np.sum(exp, axis=1, keepdims=True)          # (N, C)

    # ------------------------------------
    # Per-sample losses
    # ------------------------------------
    eps = 1e-12
    p_correct = probs[np.arange(N), targets] + eps  # (N,)
    per_sample_loss = -np.log(p_correct, dtype=np.float32)    # (N,)

    if weight is not None:
        weight = np.asarray(weight, dtype=np.float32)
        if weight.shape != (C,):
            raise ValueError(
                f"weight must have shape (C,), got {weight.shape}, C={C}"
            )
        sample_weights = weight[targets]                    # (N,)
        per_sample_loss = per_sample_loss * sample_weights  # weighted losses
    else:
        sample_weights = np.ones_like(per_sample_loss, dtype=np.float32)

    # ------------------------------------
    # Reduction
    # ------------------------------------
    if reduction_enum is ScorchReduction.NONE:
        loss: float | np.ndarray = per_sample_loss.astype(np.float32)
        normalizer = np.ones_like(sample_weights, dtype=np.float32)  # no scaling
        denom = 1.0  # not used
    else:
        if weight is not None:
            # PyTorch-style: mean = sum(weighted_losses) / sum(sample_weights)
            weight_sum = float(np.sum(sample_weights))
            if weight_sum <= 0.0:
                # Degenerate; avoid div-zero
                mean_loss = 0.0
                denom = 1.0
            else:
                mean_loss = float(np.sum(per_sample_loss) / weight_sum)
                denom = weight_sum
        else:
            mean_loss = float(per_sample_loss.mean())
            denom = float(N)

        if reduction_enum is ScorchReduction.MEAN:
            loss = mean_loss
        elif reduction_enum is ScorchReduction.SUM:
            loss = float(np.sum(per_sample_loss))
            denom = 1.0  # sum => no global scale in grad
        else:
            raise RuntimeError(f"Unknown reduction: {reduction_enum}")

        # For gradient scaling below
        normalizer = np.full_like(sample_weights, denom, dtype=np.float32)

    # ------------------------------------
    # Gradient w.r.t. logits
    # ------------------------------------
    grad_logits = probs.copy()                              # (N, C)
    grad_logits[np.arange(N), targets] -= 1.0               # probs - one_hot

    # Apply sample weights (if any)
    grad_logits *= sample_weights[:, None]                  # (N, C)

    # Apply reduction scaling:
    # - NONE: normalizer is 1 -> no scaling
    # - MEAN: divide by sum(weights) or N
    # - SUM:  denom=1 -> no extra scaling
    grad_logits /= normalizer[:, None]

    return loss, grad_logits.astype(np.float32)


# -------------------------------------------------------------------
# Module: ScorchCrossEntropyLoss
# -------------------------------------------------------------------

class ScorchCrossEntropyLoss(ScorchWeightedLoss):
    """
    Module wrapper around cross_entropy_with_grad, like torch.nn.CrossEntropyLoss
    (but WITHOUT autograd integration).

    Usage pattern 1 (recommended for training):
        loss, grad_logits = cross_entropy_with_grad(logits, targets, weight, reduction)
        # use grad_logits for backprop via ScorchSequential.backward(...)

    Usage pattern 2 (for metric-style use):
        criterion = ScorchCrossEntropyLoss()
        loss_value = criterion.forward(logits, targets)
    """

    def __init__(
        self,
        weight: Optional[np.ndarray] = None,
        reduction: ReductionLike = ScorchReduction.MEAN,
    ) -> None:
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, input: np.ndarray, target: np.ndarray):
        """
        Compute the cross-entropy loss value (no gradient).

        For gradient + loss together, call cross_entropy_with_grad(...)
        directly instead of going through this module.
        """
        loss, _ = cross_entropy_with_grad(
            logits=input,
            targets=target,
            weight=self.weight,
            reduction=self.reduction,
        )
        return loss
