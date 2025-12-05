# scorch/annotation_loader.py
import json
from pathlib import Path
from filesystem.file_io import FileIO
from labels.image_annotation_document import ImageAnnotationDocument


def load_annotation_document(path: str | Path) -> ImageAnnotationDocument:
    """
    Load an annotation JSON file from an absolute or project-local path.
    """
    #p = Path(path).resolve()

    p = FileIO.local_file(name="////testing/proto_cells_test_000_annotations.json").resolve()

    raw_bytes = FileIO.load(p)
    data = json.loads(raw_bytes.decode("utf-8"))

    return ImageAnnotationDocument.from_json(data)
