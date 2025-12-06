# scorch/data_loader.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from filesystem.file_io import FileIO
from labels.image_annotation_document import ImageAnnotationDocument
from .annotation_loader import load_annotation_document, AnnotationLoadError


@dataclass
class AnnotationImagePair:
    """
    Holds a full annotation document and the matching image path on disk.
    """
    document: ImageAnnotationDocument
    image_path: Path

# scorch/data_loader.py  (continued)

class DataLoader:
    """
    Generic loader that:
      - finds all annotation JSON files via FileIO.get_all_files_local
      - loads each into an ImageAnnotationDocument
      - verifies the corresponding PNG image exists (name + ".png")
      - yields AnnotationImagePair objects
    """

    def __init__(self, annotations_subdir: str, images_subdir: str | None = None):
        """
        annotations_subdir: project-local subdirectory where *_annotations.json live
        images_subdir:      project-local subdirectory where PNG images live
                            (defaults to the same as annotations_subdir)
        """
        self.annotations_subdir = annotations_subdir
        self.images_subdir = images_subdir or annotations_subdir

        # --- Discover annotation files ---
        all_files: List[Path] = FileIO.get_all_files_local(
            subdirectory=self.annotations_subdir
        )

        # Keep only files that look like annotation JSONs.
        self.annotation_files: List[Path] = sorted(
            f for f in all_files
            if f.suffix == ".json" and f.name.endswith("_annotations.json")
        )

    def __len__(self) -> int:
        return len(self.annotation_files)

    def __iter__(self):
        seen = set()

        for ann_path in self.annotation_files:
            try:
                doc = load_annotation_document(ann_path)
            except AnnotationLoadError as e:
                print(f"[DataLoader] Skipping corrupt file: {ann_path}\n  {e}")
                continue  # skip but keep going

            name = doc.name

            if name in seen:
                print("DUPE NAME??? ", name)
                continue
            seen.add(name)

            # build image path
            img_rel = Path(self.images_subdir) / f"{name}.png"
            img_path = FileIO.local_file(name=str(img_rel)).resolve()

            if not img_path.exists():
                print(f"[DataLoader] Missing PNG for '{name}', skipping.")
                continue

            yield AnnotationImagePair(document=doc, image_path=img_path)

