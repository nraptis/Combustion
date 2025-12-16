# data_label.py
from __future__ import annotations

from typing import Any, Dict, List, Iterable, Optional
from image.bitmap import Bitmap
from image.rgba import RGBA
from labels.pixel_bag import PixelBag
from image.pooling_mode import PoolingMode
from image.convolve_kernel_alignment import ConvolveKernelAlignment
from image.convolve_padding_mode import (
    ConvolvePaddingMode,
    ConvolvePaddingSame,
    ConvolvePaddingValid,
    ConvolvePaddingOffsetSame,
    ConvolvePaddingOffsetValid,
)

class DataLabel:
    """
    Associates a string name with a PixelBag.

    JSON format:
        {
            "name": "some_label_name",
            "pixels": [ { "y": ..., "x_start": ..., "x_end": ... }, ... ]
        }
    """

    def __init__(self, name: str, pixel_bag: Optional[PixelBag] = None) -> None:
        self.name: str = name
        self.pixel_bag: PixelBag = pixel_bag if pixel_bag is not None else PixelBag()

    def clear(self) -> None:
        self.pixel_bag.clear()

    def add(self, x: int, y: int) -> None:
        self.pixel_bag.add(x, y)

    def remove(self, x: int, y: int) -> None:
        self.pixel_bag.remove(x, y)

    def contains(self, x: int, y: int) -> bool:
        return self.pixel_bag.contains(x, y)
    
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
    
    def __repr__(self) -> str:
        return f'DataLabel(name="{self.name}", bag={self.pixel_bag})'

class DataLabelCollection:

    def __init__(self, labels: List[DataLabel] | None = None) -> None:
        self.labels: List[DataLabel] = labels or []

    def add_label(self, label: DataLabel) -> None:
        self.labels.append(label)

    def remove_label(self, label: DataLabel) -> None:
        try:
            self.labels.remove(label)
        except ValueError:
            # label not in list; ignore
            pass

    def get_labels_by_name(self, name: str) -> List[DataLabel]:
        return [lbl for lbl in self.labels if lbl.name == name]

    def first_label(self, name: str) -> DataLabel | None:
        for lbl in self.labels:
            if lbl.name == name:
                return lbl
        return None
    
    def to_json(self) -> List[Dict[str, Any]]:
        return [label.to_json() for label in self.labels]

    @staticmethod
    def from_json(data: List[Dict[str, Any]]) -> "DataLabelCollection":
        labels = [DataLabel.from_json(item) for item in data]
        return DataLabelCollection(labels=labels)
    
    def to_bitmap(
        self,
        image_width: int,
        image_height: int,
        data_label_color: Optional[RGBA] = None,
        background_color: Optional[RGBA] = None,
    ) -> Optional[Bitmap]:
        """
        Render the entire collection into a single bitmap.

        - Floods background once
        - Stamps each label's PixelBag using PixelBag.stamp_bitmap (RGB-only, alpha unchanged)
        - Safe bounds (handled by stamp_bitmap)
        """
        W = int(image_width)
        H = int(image_height)
        if W <= 0 or H <= 0:
            return None

        bg = background_color if isinstance(background_color, RGBA) else RGBA(0, 0, 0, 0)

        bmp = Bitmap(W, H)
        bmp.flood(bg)
        
        for label in self.labels:
            bag = label.pixel_bag
            if bag is None or len(bag) == 0:
                continue
            bag.stamp_bitmap(bmp, data_label_color)

        return bmp


    def transformed_with_convolve(
        self,
        mask: List[List[float]],
        image_width: int,
        image_height: int,
        offset_x: int = 0,
        offset_y: int = 0,
        stride_h: int = 1,
        stride_v: int = 1,
        dilation_h: int = 1,
        dilation_v: int = 1,
        kernel_alignment: ConvolveKernelAlignment = ConvolveKernelAlignment.CENTER,
        padding_mode: ConvolvePaddingMode = ConvolvePaddingValid(),
    ) -> "DataLabelCollection":
        out = DataLabelCollection()
        for label in self.labels:
            new_bag = PixelBag.transformed_with_convolve(
                pixel_bag=label.pixel_bag,
                mask=mask,
                image_width=image_width,
                image_height=image_height,
                offset_x=offset_x,   # ✅
                offset_y=offset_y,   # ✅
                stride_h=stride_h,
                stride_v=stride_v,
                dilation_h=dilation_h,
                dilation_v=dilation_v,
                kernel_alignment=kernel_alignment,
                padding_mode=padding_mode,
            )
            if new_bag is None or len(new_bag) == 0:
                continue
            out.add_label(DataLabel(name=label.name, pixel_bag=new_bag))
        return out

    
    def __len__(self):
        return len(self.labels)

    def __iter__(self) -> Iterable[DataLabel]:
        return iter(self.labels)

    def _sorted_labels(self) -> List[DataLabel]:
        if not self.labels:
            return []
        def sort_key(label: DataLabel):
            name = label.name
            summary = label.pixel_bag.summary()
            median = summary.get("median")
            if median is None:
                my = 10**9
                mx = 10**9
            else:
                mx, my = median
            return (name, my, mx)
        return sorted(self.labels, key=sort_key)
    
    def __repr__(self) -> str:
        count = len(self.labels)
        if count == 0:
            return "DataLabelCollection(count=0)"
        lines: List[str] = [f"DataLabelCollection(count={count}):"]
        for label in self._sorted_labels():
            lines.append(f"    {label!r}")
        return "\n".join(lines)