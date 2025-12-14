# medicool_annotation_mask_exporter.py

from __future__ import annotations
from pathlib import Path
from typing import List
from labels.data_label import DataLabel
from labels.image_annotation_document import ImageAnnotationDocument
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
          list of exported mask filenames (with .png extension)
        """
        exported_mask_names: List[str] = []
        success = 0
        failure = 0

        def num_str(i: int) -> str:
            return f"{i:0{minimum_digit_count}d}"

        WHITE = RGBA(255, 255, 255, 255)
        BLACK = RGBA(0, 0, 0, 255)

        for source_label_file_name in source_label_file_names:
            try:
                document = ImageAnnotationDocument.from_local_file(
                    subdirectory=source_label_subdir,
                    name=source_label_file_name,
                    extension=None,
                )

                W = int(document.width)
                H = int(document.height)
                if W <= 0 or H <= 0:
                    raise ValueError(f"Document has invalid dimensions ({W},{H})")

                labels: List[DataLabel] = document.data.labels
                source_stem = Path(source_label_file_name).stem  # no extension

                for export_index, label in enumerate(labels):
                    try:
                        # White background
                        bitmap = Bitmap(W, H)
                        bitmap.flood(WHITE)

                        bag = getattr(label, "pixel_bag", None)
                        if bag is None or len(bag) == 0:
                            # Export an all-white mask for empty labels (still useful for debugging)
                            pass
                        else:
                            # Stamp label pixels as black
                            for (x0, y0) in bag:
                                x = int(x0)
                                y = int(y0)
                                if 0 <= x < W and 0 <= y < H:
                                    px = bitmap.rgba[x][y]
                                    px.ri = BLACK.ri
                                    px.gi = BLACK.gi
                                    px.bi = BLACK.bi
                                    px.ai = BLACK.ai

                        export_name_no_ext = (
                            f"{source_stem}{destination_mask_suffix}_{num_str(export_index)}"
                        )

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
                        print(
                            f"❌ Mask export failed (label {export_index}) "
                            f"for {source_label_file_name}: {e_label}"
                        )

            except Exception as e_file:
                failure += 1
                print(f"❌ Label file failed: {source_label_file_name}: {e_file}")

        print(f"✅ Mask export complete: {success} succeeded, {failure} failed.")
        return exported_mask_names
