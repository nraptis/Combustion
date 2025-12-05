# pixel_bag_run_length_stripe.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class PixelBagRunLengthStripe:
    """
    Represents a horizontal run of pixels at a fixed y:
        x in [x_start, x_end] inclusive.
    """
    y: int
    x_start: int
    x_end: int

    def to_json(self) -> Dict[str, Any]:
        """
        Serialize this stripe to a JSON-compatible dict.
        """
        return {
            "y": int(self.y),
            "x_start": int(self.x_start),
            "x_end": int(self.x_end),
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "PixelBagRunLengthStripe":
        """
        Deserialize a stripe from a JSON-compatible dict.
        Expects keys: "y", "x_start", "x_end".
        """
        return PixelBagRunLengthStripe(
            y=int(data["y"]),
            x_start=int(data["x_start"]),
            x_end=int(data["x_end"]),
        )

    # --------------------------------------------------
    # Debug / printing helpers
    # --------------------------------------------------
    def repr_str(self, indent: int = 0) -> str:
        """
        Return a single-line string representation, with leading tabs
        based on indent depth.
        """
        prefix = "\t" * max(indent, 0)
        return f"{prefix}(y={self.y}, x_start={self.x_start}, x_end={self.x_end})"

    def __repr__(self) -> str:
        """
        Default repr with no indentation.
        """
        return self.repr_str(indent=0)
