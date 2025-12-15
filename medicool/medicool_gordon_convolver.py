# medicool_gordon_convolver.py

from __future__ import annotations

from typing import List, Tuple

from filesystem.file_utils import FileUtils

from labels.pixel_bag import PixelBag
from labels.data_label import DataLabel
from labels.image_annotation_document import ImageAnnotationDocument

from image_tools.mask_loader import load_mask_white_xy_weights
from image.bitmap import Bitmap
from image.convolve_padding_mode import ConvolvePaddingMode

from image.convolve_padding_mode import (
    ConvolvePaddingMode,
    ConvolvePaddingSame,
    ConvolvePaddingValid,
    ConvolvePaddingOffsetSame,
    ConvolvePaddingOffsetValid,
)

class MedicoolGordonConvolver:

    # NOTE: trim no longer used by Bitmap.convolve (Torch-like SAME/VALID now handles borders)
    TRIM_H = 16
    TRIM_V = 32  # keep this different for verify test (still used by label section for now)

    KERNEL_SUBDIR = "images"
    KERNEL_FILE = "good_egg_kernel_red_5_9.png"
    #KERNEL_FILE = "stripe_kernel_red_7_1.png"

    OUTPUT_SUFFIX = "_gor"  # gordon chase, street farmer commando (extaordinary demon slayer)

    # ------------------------------------------------------------
    # New Torch-ish convolve params (spicy defaults)
    # ------------------------------------------------------------
    STRIDE_H = 1
    STRIDE_V = 2
    
    DILATION_H = 1
    DILATION_V = 1

    # SAME uses padding; VALID assumes no padding.
    PADDING_MODE = ConvolvePaddingOffsetValid(60, 60)

    # Offsets: only Torch 1:1 match when both are 0
    OFFSET_X = 0
    OFFSET_Y = 0

    # Padding fill to match Torch (zeros everywhere). If your Bitmap expects opaque,
    # change this in Bitmap.convolve, not here.
    # (Convolver doesn't need explicit padding value; Bitmap handles it.)

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
                bmp = Bitmap.with_local_image(
                    subdirectory=source_image_subdir,
                    name=source_image_file_name,
                    extension=None,
                )

                convolved = bmp.convolve(
                    mask=mask,
                    offset_x=cls.OFFSET_X,
                    offset_y=cls.OFFSET_Y,
                    stride_h=cls.STRIDE_H,
                    stride_v=cls.STRIDE_V,
                    dilation_h=cls.DILATION_H,
                    dilation_v=cls.DILATION_V,
                    padding_mode=cls.PADDING_MODE,
                )

                new_file_name = FileUtils.append_file_suffix(source_image_file_name, cls.OUTPUT_SUFFIX)

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
        # NOTE: This label logic is still TRIM-based. Since convolve no longer trims,
        # you'll likely change this next once you decide how labels should transform
        # under SAME/VALID + stride/dilation/offset.
        for source_label_file_name in source_label_file_names:
            try:
                document = ImageAnnotationDocument.from_local_file(
                    subdirectory=source_label_subdir,
                    name=source_label_file_name,
                    extension=None,
                )

                new_w = document.width - (cls.TRIM_H * 2)
                new_h = document.height - (cls.TRIM_V * 2)
                if new_w <= 0 or new_h <= 0:
                    raise ValueError(
                        f"Trim produced non-positive size: ({new_w},{new_h}) "
                        f"from ({document.width},{document.height})"
                    )

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

        # Helpful run summary of convolve params
        print(
            "✅ GordonConvolver convolve params: "
            f"padding_mode={Bitmap.padding_mode_string(cls.PADDING_MODE)} "
            f"stride=({cls.STRIDE_H},{cls.STRIDE_V}) "
            f"dilation=({cls.DILATION_H},{cls.DILATION_V}) "
            f"offset=({cls.OFFSET_X},{cls.OFFSET_Y}) "
            f"kernel={cls.KERNEL_FILE}"
        )

        return out_image_files, out_label_files
