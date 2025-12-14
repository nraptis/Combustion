# medicool_gordon_convolver.py

from __future__ import annotations

from typing import List, Tuple

from filesystem.file_utils import FileUtils

from labels.pixel_bag import PixelBag
from labels.data_label import DataLabel
from labels.image_annotation_document import ImageAnnotationDocument

from image_tools.mask_loader import load_mask_white_xy_weights
from image.bitmap import Bitmap

class MedicoolGordonConvolver:

    TRIM_H = 16
    TRIM_V = 32  # keep this different for verify test

    KERNEL_SUBDIR = "images"
    #KERNEL_FILE = "good_egg_kernel_red_5_9.png"
    KERNEL_FILE = "stripe_kernel_red_7_1.png"
    
    OUTPUT_SUFFIX = "_gor"  # gordon chase, street farmer commando (extaordinary demon slayer)

    @classmethod
    def execute(
        cls,
        source_image_subdir: str,
        source_image_file_names: List[str],
        source_label_subdir: str,
        source_label_file_names: List[str],
        destination_image_subdir: str,
        destination_label_subdir: str,
    ) -> Tuple[List[str], List[str]]:
        
        out_image_files: List[str] = []
        out_label_files: List[str] = []
        failed_images: List[str] = []
        failed_labels: List[str] = []

        # ------------------------------------------------------------
        # Load mask
        # ------------------------------------------------------------
        try:
            mask = load_mask_white_xy_weights(cls.KERNEL_SUBDIR, cls.KERNEL_FILE)
            if mask is None or len(mask) == 0:
                raise ValueError("mask loader returned empty mask")
        except Exception as e:
            print(f"❌ Kernel mask load failed: {cls.KERNEL_FILE} ({e})")
            return out_image_files, out_label_files

        # ------------------------------------------------------------
        # Images: convolve -> save
        # ------------------------------------------------------------
        for source_image_file_name in source_image_file_names:
            try:
                # load bitmap from local file scheme
                bmp = Bitmap.with_local_image(
                    subdirectory=source_image_subdir,
                    name=source_image_file_name,
                    extension=None,
                )

                convolved = bmp.convolve(
                    mask=mask,
                    trim_h=cls.TRIM_H,
                    trim_v=cls.TRIM_V,
                    offset_x=0,
                    offset_y=0
                )

                new_file_name = FileUtils.append_file_suffix(source_image_file_name, cls.OUTPUT_SUFFIX)

                # FileUtils.save_local_bitmap wants name WITHOUT extension (it appends extension)
                # but your new_file_name might already include ".png".
                # So we save using the exact filename by passing extension=None and name including ext.
                FileUtils.save_local_bitmap(
                    convolved,
                    subdirectory=destination_image_subdir,
                    name=new_file_name,
                    extension=None,
                )

                out_image_files.append(new_file_name)

            except Exception as e:
                failed_images.append(source_image_file_name)
                print(f"❌ Image failed: {source_image_file_name} ({e})")

        # ------------------------------------------------------------
        # Labels: shift + clip -> save
        # ------------------------------------------------------------
        for source_label_file_name in source_label_file_names:
            try:
                # from_local_file expects 'name' without extension when extension is provided.
                # Here we pass extension=None, so name can include ".json" and still work.
                document = ImageAnnotationDocument.from_local_file(
                    subdirectory=source_label_subdir,
                    name=source_label_file_name,
                    extension=None,
                )

                new_w = document.width - (cls.TRIM_H * 2)
                new_h = document.height - (cls.TRIM_V * 2)
                if new_w <= 0 or new_h <= 0:
                    raise ValueError(f"Trim produced non-positive size: ({new_w},{new_h}) from ({document.width},{document.height})")

                new_document = ImageAnnotationDocument(
                    name=document.name,
                    width=new_w,
                    height=new_h,
                )

                for label in document.data.labels:
                    old_bag = label.pixel_bag
                    new_bag = PixelBag()

                    for (x, y) in old_bag.pixels:
                        nx = int(x) - cls.TRIM_H
                        ny = int(y) - cls.TRIM_V
                        if 0 <= nx < new_w and 0 <= ny < new_h:
                            new_bag.add(nx, ny)

                    if len(new_bag) > 0:
                        new_document.data.add_label(DataLabel(label.name, new_bag))

                new_file_name = FileUtils.append_file_suffix(source_label_file_name, cls.OUTPUT_SUFFIX)

                # Save JSON. FileUtils.save_local_json expects name without extension if extension provided.
                # We want to preserve existing extension if present; append_file_suffix preserves it.
                FileUtils.save_local_json(
                    new_document.to_json(),
                    subdirectory=destination_label_subdir,
                    name=new_file_name,
                    extension=None,
                )

                out_label_files.append(new_file_name)

            except Exception as e:
                failed_labels.append(source_label_file_name)
                print(f"❌ Label failed: {source_label_file_name} ({e})")

        # ------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------
        print(f"✅ GordonConvolver images: {len(out_image_files)} ok, {len(failed_images)} failed.")
        if failed_images:
            print("   Failed images:")
            for f in failed_images:
                print(f"   - {f}")

        print(f"✅ GordonConvolver labels: {len(out_label_files)} ok, {len(failed_labels)} failed.")
        if failed_labels:
            print("   Failed labels:")
            for f in failed_labels:
                print(f"   - {f}")

        return out_image_files, out_label_files
