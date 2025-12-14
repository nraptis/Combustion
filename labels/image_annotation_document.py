# image_annotation_document.py

from __future__ import annotations

from typing import Any, Dict, List
from labels.data_label_collection import DataLabelCollection
from filesystem.file_utils import FileUtils


class ImageAnnotationDocument:
    """
    Top-level container for annotations for a single image.
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

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "ImageAnnotationDocument":
        name = data.get("name", "")
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))

        labels_raw = data.get("labels", []) or []
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
        """
        Load a document from a local JSON file using FileUtils.load_local_json().
        """
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
