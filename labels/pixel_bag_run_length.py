# pixel_bag_run_length.py
from __future__ import annotations
from typing import Any, Iterable, List

from labels.pixel_bag_run_length_stripe import PixelBagRunLengthStripe

class PixelBagRunLength:
    """
    Run-length representation of a PixelBag:
    a list of horizontal stripes.
    """

    def __init__(self, stripes: List[PixelBagRunLengthStripe] | None = None) -> None:
        self.stripes: List[PixelBagRunLengthStripe] = stripes or []

    # --------------------------------------------------
    # JSON serialization
    # --------------------------------------------------
    def to_json(self) -> List[Any]:
        return [stripe.to_json() for stripe in self.stripes]

    @staticmethod
    def from_json(data: List[Any]) -> "PixelBagRunLength":
        stripes = [PixelBagRunLengthStripe.from_json(item) for item in data]
        return PixelBagRunLength(stripes=stripes)

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------
    def add_stripe(self, stripe: PixelBagRunLengthStripe) -> None:
        self.stripes.append(stripe)

    def __len__(self) -> int:
        return len(self.stripes)

    def __iter__(self):
        return iter(self.stripes)

    # -------- central sorting helper --------
    @staticmethod
    def sorted_stripes(
        stripes: Iterable[PixelBagRunLengthStripe],
    ) -> List[PixelBagRunLengthStripe]:
        return sorted(stripes, key=lambda s: (s.y, s.x_start))

    def sorted(self) -> List[PixelBagRunLengthStripe]:
        return PixelBagRunLength.sorted_stripes(self.stripes)

    # --------------------------------------------------
    # One-line summary repr
    # --------------------------------------------------
    def __repr__(self) -> str:
        """
        Compact one-line summary:
        PixelBagRunLength(count=3, stripes=[Stripe(y=10,3→5), Stripe(y=11,7→7)])
        """
        if not self.stripes:
            return "PixelBagRunLength(count=0, stripes=[])"

        parts = []
        for s in self.sorted():
            parts.append(f"Stripe(y={s.y},{s.x_start}→{s.x_end})")

        stripes_str = ", ".join(parts)
        return f"PixelBagRunLength(count={len(self.stripes)}, stripes=[{stripes_str}])"
