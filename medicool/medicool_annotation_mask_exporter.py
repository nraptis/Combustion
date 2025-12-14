# medicool_annotation_mask_exporter.py

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
from labels.pixel_bag import PixelBag
from labels.data_label import DataLabel
from labels.data_label_collection import DataLabelCollection
from labels.image_annotation_document import ImageAnnotationDocument
from filesystem.file_io import FileIO
from filesystem.file_utils import FileUtils
from image.bitmap import Bitmap
from image.rgba import RGBA

class MedicoolAnnotationMaskExporter:

    @classmethod
    def execute(
        cls,
        source_label_subdir: str,
        source_label_file_names: List[str],
        destination_mask_subdir: str,
        destination_mask_suffix: str,
        minimum_digit_count: int = 3,
    ) -> List[str]:
        """
        For each label file:
          - loads JSON -> ImageAnnotationDocument
          - exports ONE mask PNG per DataLabel
          - white background, black pixels for the label's PixelBag

        Output name format:
          <source_stem><destination_mask_suffix>_<NNN>.png
        Returns:
          (exported_mask_names, success_count, failure_count)
        """
        exported_mask_names: List[str] = []
        success = 0
        failure = 0

        def num_str(i: int) -> str:
            return f"{i:0{minimum_digit_count}d}"

        for source_label_file_name in source_label_file_names:
            try:
                document = ImageAnnotationDocument.from_local_file(subdirectory=source_label_subdir,
                    name=source_label_file_name,
                    extension=None)

                labels: List[DataLabel] = document.data.labels

                # stem without extension (proto_cells_train_000_annotations)
                source_stem = Path(source_label_file_name).stem

                for export_index, label in enumerate(labels):
                    try:
                        # White background
                        bitmap = Bitmap(document.width, document.height)
                        bitmap.flood(RGBA(0, 0, 0, 255))

                        # Stamp label pixels as black
                        bag = label.pixel_bag
                        for (x, y) in bag:
                            if 0 <= x < bitmap.width and 0 <= y < bitmap.height:
                                px = bitmap.rgba[x][y]
                                px.ri = 255
                                px.gi = 255
                                px.bi = 255
                                px.ai = 255

                        # Build export name (NO extension here)
                        export_name_no_ext = (
                            f"{source_stem}{destination_mask_suffix}_{num_str(export_index)}"
                        )

                        # Save as png
                        FileUtils.save_local_bitmap(
                            bitmap,
                            subdirectory=destination_mask_subdir,
                            name=export_name_no_ext,
                            extension="png",
                        )

                        exported_mask_names.append(export_name_no_ext + ".png")
                        success += 1

                    except Exception as e_label:
                        failure += 1
                        print(f"❌ Mask export failed (label {export_index}) for {source_label_file_name}: {e_label}")

            except Exception as e_file:
                # If the whole file fails to load/parse, count it as one failure
                failure += 1
                print(f"❌ Label file failed: {source_label_file_name}: {e_file}")

        print(f"✅ Mask export complete: {success} succeeded, {failure} failed.")
        return exported_mask_names
