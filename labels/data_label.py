# data_label.py
from __future__ import annotations

from typing import Any, Dict
from labels.pixel_bag import PixelBag

class DataLabel:
    """
    Associates a string name with a PixelBag.

    JSON format:

        {
            "name": "some_label_name",
            "pixels": [ { "y": ..., "x_start": ..., "x_end": ... }, ... ]
        }
    """

    def __init__(self, name: str, pixel_bag: PixelBag | None = None) -> None:
        self.name: str = name
        self.pixel_bag: PixelBag = pixel_bag if pixel_bag is not None else PixelBag()


    # --------------------------------------------------
    # PixelBag passthrough wrappers
    # --------------------------------------------------
    def clear(self) -> None:
        self.pixel_bag.clear()

    def add(self, x: int, y: int) -> None:
        self.pixel_bag.add(x, y)

    def remove(self, x: int, y: int) -> None:
        self.pixel_bag.remove(x, y)

    def contains(self, x: int, y: int) -> bool:
        return self.pixel_bag.contains(x, y)

    # --------------------------------------------------
    # JSON serialization
    # --------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pixels": self.pixel_bag.to_json(),
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "DataLabel":
        name = data.get("name", "")
        pixels_json = data.get("pixels", [])
        bag = PixelBag.from_json(pixels_json)
        return DataLabel(name=name, pixel_bag=bag)

    # --------------------------------------------------
    # Utility / debug printing (summary style)
    # --------------------------------------------------
    def __repr__(self) -> str:
        """
        One-line compact summary, e.g.:

            DataLabel(name="lymph",
                      bag=PixelBag(count=100, median=(15,35), size=(10,10)))

        If bag is empty:

            DataLabel(name="lymph", bag=PixelBag(count=0))
        """
        return f'DataLabel(name="{self.name}", bag={self.pixel_bag})'
