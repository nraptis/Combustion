# pixel_bag.py
from __future__ import annotations
from typing import Any, List
from labels.pixel_bag_run_length import PixelBagRunLength
from labels.pixel_bag_run_length_stripe import PixelBagRunLengthStripe

class PixelBag:
    """
    Stores a set of (x, y) integer pixel positions.
    Does not store duplicates (internally a set),
    but add/remove semantics allow double-add and remove-nonexistent
    without raising exceptions.
    """

    def __init__(self) -> None:
        self._set = set()   # stores (x, y)

    # --------------------------------------------------
    # Basic operations
    # --------------------------------------------------
    def clear(self) -> None:
        self._set.clear()

    def add(self, x: int, y: int) -> None:
        """Add a pixel. Adding an existing pixel is allowed and ignored."""
        self._set.add((int(x), int(y)))

    def remove(self, x: int, y: int) -> None:
        """Remove a pixel. Removing a missing pixel is allowed and ignored."""
        self._set.discard((int(x), int(y)))  # discard() never raises

    def contains(self, x: int, y: int) -> bool:
        """Check if a pixel exists in the bag."""
        return (int(x), int(y)) in self._set

    # --------------------------------------------------
    # Bounding box helpers
    # --------------------------------------------------
    @property
    def xmin(self):
        if not self._set:
            return None
        return min(px for (px, _) in self._set)

    @property
    def xmax(self):
        if not self._set:
            return None
        return max(px for (px, _) in self._set)

    @property
    def ymin(self):
        if not self._set:
            return None
        return min(py for (_, py) in self._set)

    @property
    def ymax(self):
        if not self._set:
            return None
        return max(py for (_, py) in self._set)

    # --------------------------------------------------
    # Ranges for easy looping
    # --------------------------------------------------
    def xrange(self):
        """
        Return range(xmin, xmax+1).
        If empty, return range(0).
        """
        if not self._set:
            return range(0)
        return range(self.xmin, self.xmax + 1)

    def yrange(self):
        """
        Return range(ymin, ymax+1).
        If empty, return range(0).
        """
        if not self._set:
            return range(0)
        return range(self.ymin, self.ymax + 1)

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------
    def __len__(self):
        return len(self._set)

    def __iter__(self):
        """Iterate over (x, y) pairs."""
        return iter(self._set)

    def summary(self) -> dict:
        """
        Return summary statistics:

            {
            "count": int,
            "median": (x_med, y_med) | None,
            "size": (width, height)
            }

        width  = xmax - xmin + 1
        height = ymax - ymin + 1
        """
        count = len(self._set)
        if count == 0:
            return {
                "count": 0,
                "median": None,
                "size": (0, 0),
            }

        xs = [x for (x, _) in self._set]
        ys = [y for (_, y) in self._set]

        xs_sorted = sorted(xs)
        ys_sorted = sorted(ys)
        mid = count // 2

        median_x = xs_sorted[mid]
        median_y = ys_sorted[mid]

        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax

        width = (xmax - xmin + 1) if xmin is not None and xmax is not None else 0
        height = (ymax - ymin + 1) if ymin is not None and ymax is not None else 0

        return {
            "count": count,
            "median": (median_x, median_y),
            "size": (width, height),
        }


    def __repr__(self):
        """
        Compact one-line summary:

            PixelBag(count=67, median=(46,77), size=(80, 111))
        """
        info = self.summary()
        count = info["count"]

        if count == 0:
            return "PixelBag(count=0)"

        mx, my = info["median"]
        w, h = info["size"]

        return (
            f"PixelBag(count={count}, "
            f"median=({mx}, {my}), "
            f"size=({w}, {h}))"
        )


    # --------------------------------------------------
    # Run-length conversion
    # --------------------------------------------------
    def to_run_length(self) -> "PixelBagRunLength":
        result = PixelBagRunLength()

        _xmin = self.xmin
        _xmax = self.xmax
        if _xmin is None or _xmax is None:
            return result

        for y in self.yrange():
            x = _xmin
            while x <= _xmax:
                if self.contains(x, y):
                    x_start = x
                    x_end = x
                    x += 1
                    while x <= _xmax and self.contains(x, y):
                        x_end = x
                        x += 1
                    result.add_stripe(PixelBagRunLengthStripe(y, x_start, x_end))
                else:
                    x += 1
        return result

    # --------------------------------------------------
    # JSON serialization
    # --------------------------------------------------
    def to_json(self) -> List[Any]:
        """
        Serialize this PixelBag to a JSON-compatible list of run-length
        stripe objects:
            [ { "y": ..., "x_start": ..., "x_end": ... }, ... ]
        """
        rle = self.to_run_length()
        return rle.to_json()

    @staticmethod
    def from_json(data: List[Any]) -> "PixelBag":
        """
        Deserialize a PixelBag from a JSON-compatible list produced
        by PixelBag.to_json().

        Reconstructs all (x, y) pixels from the run-length stripes.
        """
        rle = PixelBagRunLength.from_json(data)
        bag = PixelBag()
        for stripe in rle.stripes:
            y = stripe.y
            for x in range(stripe.x_start, stripe.x_end + 1):
                bag.add(x, y)
        return bag
