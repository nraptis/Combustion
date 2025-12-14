# pixel_bag.py
from __future__ import annotations
from typing import Any, List, Optional
from labels.pixel_bag_run_length import PixelBagRunLength
from labels.pixel_bag_run_length_stripe import PixelBagRunLengthStripe
from image.bitmap import Bitmap

class PixelBag:
    """
    Stores a set of (x, y) integer pixel positions.
    Does not store duplicates (internally a set),
    but add/remove semantics allow double-add and remove-nonexistent
    without raising exceptions.
    """

    def __init__(self) -> None:
        self._set = set()   # stores (x, y)

    @property
    def pixels(self) -> List[tuple[int, int]]:
        """
        Return all pixels in this bag as a list of (x, y) tuples.
        Always returns a list (never None).
        """
        return list(self._set)

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
    
    @property
    def frame(self):
        if not self._set:
            return (0, 0, 0, 0)

        xr = self.xrange()
        yr = self.yrange()
        
        width  = xr.stop - xr.start
        height = yr.stop - yr.start

        return (xr.start, yr.start, width, height)

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
    
    @classmethod
    def transforming_with_convolve(
        cls,
        pixel_bag: Optional["PixelBag"],
        mask: List[List[float]],
        image_width: int,
        image_height: int,
        trim_h: int,
        trim_v: int,
        offset_x: int = 0,
        offset_y: int = 0) -> Optional["PixelBag"]:

        if pixel_bag is None or len(pixel_bag) == 0:
            return None

        W = int(image_width)
        H = int(image_height)

        # STRICT: will raise if invalid / OOB / non-positive output
        start_x, start_y, out_w, out_h = Bitmap.convolve_frame(
            W, H, mask, trim_h, trim_v, offset_x, offset_y
        )

        result = PixelBag()

        # PixelBag is in SOURCE coords. Convolved bitmap is in OUTPUT coords.
        # Output (0,0) corresponds to source base at (start_x, start_y).
        for (x, y) in pixel_bag:
            nx = int(x) - int(start_x)
            ny = int(y) - int(start_y)
            if 0 <= nx < out_w and 0 <= ny < out_h:
                result.add(nx, ny)

        return result if len(result) > 0 else None
        
    @classmethod
    def transforming_with_pool(
        cls,
        pixel_bag: Optional["PixelBag"],
        image_width: int,
        image_height: int,
        kernel_width: int = 2,
        kernel_height: int = 2,
        stride_h: int = 0,
        stride_v: int = 0) -> Optional["PixelBag"]:

        if pixel_bag is None or len(pixel_bag) == 0:
            return None

        W = int(image_width)
        H = int(image_height)
        kw = int(kernel_width)
        kh = int(kernel_height)
        sh = int(stride_h) if int(stride_h) > 0 else kw
        sv = int(stride_v) if int(stride_v) > 0 else kh

        # STRICT: will raise if invalid / non-positive output
        _, _, out_w, out_h = Bitmap.pool_frame(W, H, kw, kh, sh, sv)

        result = PixelBag()

        total = kw * kh
        # "at least half" (ties pass). If you want strict majority: total//2 + 1
        thresh = (total + 1) // 2

        for oy in range(out_h):
            y0 = oy * sv
            y1 = y0 + kh
            for ox in range(out_w):
                x0 = ox * sh
                x1 = x0 + kw

                inside = 0
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        if pixel_bag.contains(x, y):
                            inside += 1
                            if inside >= thresh:
                                break
                    if inside >= thresh:
                        break

                if inside >= thresh:
                    # pooled coords (matches pooled bitmap space)
                    result.add(ox, oy)

        return result if len(result) > 0 else None
    
    @classmethod
    def transforming_with_pool_experimental(
        cls,
        pixel_bag: Optional["PixelBag"],
        image_width: int,
        image_height: int,
        kernel_width: int = 2,
        kernel_height: int = 2,
        stride_h: int = 0,
        stride_v: int = 0,
    ) -> Optional["PixelBag"]:

        if pixel_bag is None or len(pixel_bag) == 0:
            return None

        W = int(image_width)
        H = int(image_height)
        kw = int(kernel_width)
        kh = int(kernel_height)
        sh = int(stride_h) if int(stride_h) > 0 else kw
        sv = int(stride_v) if int(stride_v) > 0 else kh

        # strict (raises if invalid)
        _, _, out_w, out_h = Bitmap.pool_frame(W, H, kw, kh, sh, sv)

        total = kw * kh
        thresh = (total + 1) // 2  # same as your brute-force version

        # votes[oy,ox] = number of "on" pixels inside that pooling window
        votes = np.zeros((out_h, out_w), dtype=np.int32)

        # integer ceil_div for possibly-negative numerators
        def ceil_div(a: int, b: int) -> int:
            return -((-a) // b)

        for (x0, y0) in pixel_bag:
            x = int(x0)
            y = int(y0)

            # skip pixels outside the image (just in case)
            if x < 0 or x >= W or y < 0 or y >= H:
                continue

            # ox range of windows that include x
            ox_min = ceil_div(x - (kw - 1), sh)
            ox_max = x // sh

            # oy range of windows that include y
            oy_min = ceil_div(y - (kh - 1), sv)
            oy_max = y // sv

            # clamp to valid output coords
            if ox_min < 0: ox_min = 0
            if oy_min < 0: oy_min = 0
            if ox_max > out_w - 1: ox_max = out_w - 1
            if oy_max > out_h - 1: oy_max = out_h - 1

            if ox_min > ox_max or oy_min > oy_max:
                continue

            votes[oy_min:oy_max+1, ox_min:ox_max+1] += 1

        result = PixelBag()
        ys, xs = np.where(votes >= thresh)
        for oy, ox in zip(ys.tolist(), xs.tolist()):
            result.add(int(ox), int(oy))

        return result if len(result) > 0 else None


