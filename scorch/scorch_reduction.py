# scorch/scorch_reduction.py
from __future__ import annotations
from enum import Enum


class ScorchReduction(str, Enum):
    """
    Enum representing how a loss should be reduced.

    - NONE  => return per-sample losses, shape (N,)
    - MEAN  => return a single scalar (average)
    - SUM   => return a single scalar (sum)

    Matches PyTorch's Reduction enum in spirit.
    """

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"

    @staticmethod
    def from_value(val: str | ScorchReduction) -> ScorchReduction:
        """
        Accepts either:
            - a ScorchReduction enum value
            - or a string ("none", "mean", "sum")

        Converts and returns a ScorchReduction.
        """
        if isinstance(val, ScorchReduction):
            return val

        if isinstance(val, str):
            val_lower = val.lower()
            for r in ScorchReduction:
                if r.value == val_lower:
                    return r

        raise ValueError(
            f"Invalid reduction: {val!r}. "
            f"Expected one of: {[r.value for r in ScorchReduction]}"
        )
