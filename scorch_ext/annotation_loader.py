# scorch_ext/annotation_loader.py
import json
from pathlib import Path
from filesystem.file_io import FileIO
from labels.image_annotation_document import ImageAnnotationDocument


class AnnotationLoadError(Exception):
    """Raised when an annotation JSON file cannot be parsed or validated."""
    pass


def load_annotation_document(path: str | Path) -> ImageAnnotationDocument:
    """
    Load an annotation JSON file safely from an absolute path.
    - We assume `path` is absolute or resolvable to absolute.
    """
    p = Path(path).resolve()

    # --- Stage 1: load raw bytes ---
    try:
        raw_bytes = FileIO.load(p)
    except Exception as e:
        raise AnnotationLoadError(f"Failed to read file: {p}") from e

    # --- Stage 2: decode UTF-8 ---
    try:
        text = raw_bytes.decode("utf-8")
    except Exception as e:
        raise AnnotationLoadError(f"Invalid UTF-8 in file: {p}") from e

    # --- Stage 3: JSON parse ---
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AnnotationLoadError(f"Malformed JSON in file: {p}") from e

    # --- Stage 4: convert to ImageAnnotationDocument ---
    try:
        return ImageAnnotationDocument.from_json(data)
    except Exception as e:
        raise AnnotationLoadError(
            f"Invalid annotation structure in file: {p}"
        ) from e
