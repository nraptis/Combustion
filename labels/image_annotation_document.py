# image_annotation_document.py

from __future__ import annotations

from typing import Any, Dict, List, Optional
from labels.data_label import DataLabel, DataLabelCollection
from filesystem.file_utils import FileUtils
from image.bitmap import Bitmap
from image.rgba import RGBA
from image.pooling_mode import PoolingMode
from image.convolve_kernel_alignment import ConvolveKernelAlignment
from image.convolve_padding_mode import (
    ConvolvePaddingMode,
    ConvolvePaddingSame,
    ConvolvePaddingValid,
    ConvolvePaddingOffsetSame,
    ConvolvePaddingOffsetValid,
)

class ImageAnnotationDocument:
    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        data: DataLabelCollection | None = None,
    ) -> None:
        self.name: str = name
        self.width: int = int(width)
        self.height: int = int(height)
        self.data: DataLabelCollection = data if data is not None else DataLabelCollection()

    @property
    def data_label_names(self) -> List[str]:
        names = {label.name for label in self.data}
        return sorted(names)

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "data_label_names": self.data_label_names,
            "labels": self.data.to_json(),
        }
    
    def to_bitmap(
        self,
        data_label_color: Optional[RGBA] = None,
        background_color: Optional[RGBA] = None,
    ) -> Optional[Bitmap]:
        """
        Render this annotation document into a bitmap of size (self.width, self.height),
        delegating to DataLabelCollection.to_bitmap().
        """
        return self.data.to_bitmap(
            image_width=self.width,
            image_height=self.height,
            data_label_color=data_label_color,
            background_color=background_color,
        )
    
    @staticmethod
    def from_json(data: Dict[str, Any]) -> "ImageAnnotationDocument":
        name = data.get("name", "")
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        labels_raw = data.get("labels") or data.get("data_labels") or data.get("annotations") or []
        dlc = DataLabelCollection.from_json(labels_raw)
        return ImageAnnotationDocument(
            name=name,
            width=width,
            height=height,
            data=dlc,
        )

    @classmethod
    def from_local_file(
        cls,
        subdirectory: str | None = None,
        name: str | None = None,
        extension: str | None = "json",
    ) -> "ImageAnnotationDocument":
        if name is None or len(str(name).strip()) == 0:
            raise ValueError("from_local_file requires a non-empty 'name'")
        data = FileUtils.load_local_json(
            subdirectory=subdirectory,
            name=name,
            extension=extension or "json",
        )
        if not isinstance(data, dict):
            raise ValueError("ImageAnnotationDocument JSON must be a dict at the top level.")
        return cls.from_json(data)

    def __repr__(self) -> str:
        label_count = len(self.data)
        header = (
            f'ImageAnnotationDocument('
            f'name="{self.name}", '
            f'size=({self.width}, {self.height}), '
            f'data_label_names={self.data_label_names}, '
            f'label_count={label_count})'
        )

        if label_count == 0:
            return header

        dlc_lines = repr(self.data).splitlines()
        indented_dlc = "\n".join("    " + line for line in dlc_lines)
        return header + ":\n" + indented_dlc
    
    def transformed_with_convolve(
        self,
        mask: List[List[float]],
        offset_x: int = 0,
        offset_y: int = 0,
        stride_h: int = 1,
        stride_v: int = 1,
        dilation_h: int = 1,
        dilation_v: int = 1,
        kernel_alignment: ConvolveKernelAlignment = ConvolveKernelAlignment.CENTER,
        padding_mode: ConvolvePaddingMode = ConvolvePaddingValid(),
    ) -> "ImageAnnotationDocument":
        ax, ay, out_w, out_h = Bitmap.convolve_frame_mask(
            image_width=self.width,
            image_height=self.height,
            mask=mask,
            offset_x=offset_x,
            offset_y=offset_y,
            stride_h=stride_h,
            stride_v=stride_v,
            dilation_h=dilation_h,
            dilation_v=dilation_v,
            kernel_alignment=kernel_alignment,
            padding_mode=padding_mode,
        )
        if out_w <= 0 or out_h <= 0:
            return ImageAnnotationDocument(name=self.name, width=0, height=0, data=DataLabelCollection())

        new_data = self.data.transformed_with_convolve(
            mask=mask,
            image_width=self.width,
            image_height=self.height,
            offset_x=offset_x,
            offset_y=offset_y,
            stride_h=stride_h,
            stride_v=stride_v,
            dilation_h=dilation_h,
            dilation_v=dilation_v,
            kernel_alignment=kernel_alignment,
            padding_mode=padding_mode,
        )
        return ImageAnnotationDocument(name=self.name, width=out_w, height=out_h, data=new_data)