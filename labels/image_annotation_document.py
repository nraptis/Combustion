# image_annotation_document.py
from __future__ import annotations

from typing import Any, Dict, List

from labels.data_label_collection import DataLabelCollection


class ImageAnnotationDocument:
    """
    Top-level container for annotations for a single image.

    Fields:
      - name:             Logical name for this document (often the JSON filename).
      - width:            Image width in pixels.
      - height:           Image height in pixels.
      - data:             DataLabelCollection holding all individual labels.
      - data_label_names: Derived list of label names present in `data`.
    """

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
        # data is conceptually non-optional; default to empty collection if None
        self.data: DataLabelCollection = data if data is not None else DataLabelCollection()

    # --------------------------------------------------
    # Derived properties
    # --------------------------------------------------
    @property
    def data_label_names(self) -> List[str]:
        """
        Return a sorted list of unique label names present in this document's data.
        """
        names = {label.name for label in self.data}
        return sorted(names)

    # --------------------------------------------------
    # JSON serialization
    # --------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        """
        Return a JSON-compatible dict representing this document.

        Layout:

        {
          "name":             str,
          "width":            int,
          "height":           int,
          "data_label_names": [str, ...],
          "labels":           [ ... DataLabel.to_json() ... ]
        }

        Note: data_label_names is derived from `data` at serialization time.
        """
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "data_label_names": self.data_label_names,
            "labels": self.data.to_json(),
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "ImageAnnotationDocument":
        """
        Parse an ImageAnnotationDocument from a JSON-compatible dict.

        data_label_names from JSON are currently ignored as a source of truth,
        since they can always be recomputed from the label data. They are
        still read (if present) to allow future validation if desired.
        """
        name = data.get("name", "")
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))

        labels_raw = data.get("labels", []) or []
        dlc = DataLabelCollection.from_json(labels_raw)

        # data_label_names = data.get("data_label_names", [])  # optional; not required

        return ImageAnnotationDocument(
            name=name,
            width=width,
            height=height,
            data=dlc,
        )

    # --------------------------------------------------
    # Debug / repr
    # --------------------------------------------------
    def __repr__(self) -> str:
        """
        Compact summary plus an indented listing of labels via
        DataLabelCollection.__repr__.

        Example:

            ImageAnnotationDocument(name="sample_01",
                                    size=(256, 256),
                                    data_label_names=['Alphacyte', 'Lymphocyte'],
                                    label_count=3):
                DataLabel(name="Alphacyte", bag=PixelBag(...))
                DataLabel(name="Lymphocyte", bag=PixelBag(...))
                ...
        """
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

        # Indent the DataLabelCollection repr by 4 spaces
        dlc_lines = repr(self.data).splitlines()
        indented_dlc = "\n".join("    " + line for line in dlc_lines)

        return header + ":\n" + indented_dlc
