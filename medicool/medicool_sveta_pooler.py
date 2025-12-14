# medicool_sveta_pooler.py

from __future__ import annotations

from typing import List, Tuple

from filesystem.file_utils import FileUtils

from labels.pixel_bag import PixelBag
from labels.data_label import DataLabel
from labels.image_annotation_document import ImageAnnotationDocument

from image.bitmap import Bitmap
from image.pooling_mode import PoolingMode

class MedicoolSvetaPooler:

    # ------------------------------------------------------------
    # Pool params
    # ------------------------------------------------------------
    POOLING_MODE = PoolingMode.MAX_PIXEL_BY_RGB_SUM

    KERNEL_WIDTH = 4
    KERNEL_HEIGHT = 4

    STRIDE_H = 3  # 0 => defaults to kernel width
    STRIDE_V = 3  # 0 => defaults to kernel height

    OUTPUT_SUFFIX = "_svet"  # sveta: the pool witch (benevolent + efficient)

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

        kw = int(cls.KERNEL_WIDTH)
        kh = int(cls.KERNEL_HEIGHT)
        sh = int(cls.STRIDE_H) if int(cls.STRIDE_H) > 0 else kw
        sv = int(cls.STRIDE_V) if int(cls.STRIDE_V) > 0 else kh

        if kw <= 0 or kh <= 0 or sh <= 0 or sv <= 0:
            raise ValueError(f"Invalid pool params kw,kh,sh,sv = ({kw},{kh},{sh},{sv})")

        # ------------------------------------------------------------
        # Images: pool -> save
        # ------------------------------------------------------------
        for source_image_file_name in source_image_file_names:
            try:
                bmp = Bitmap.with_local_image(
                    subdirectory=source_image_subdir,
                    name=source_image_file_name,
                    extension=None,
                )

                pooled = bmp.pool(
                    kernel_width=kw,
                    kernel_height=kh,
                    stride_h=sh,
                    stride_v=sv,
                    mode=cls.POOLING_MODE,
                )

                new_file_name = FileUtils.append_file_suffix(source_image_file_name, cls.OUTPUT_SUFFIX)

                FileUtils.save_local_bitmap(
                    pooled,
                    subdirectory=destination_image_subdir,
                    name=new_file_name,
                    extension=None,
                )

                out_image_files.append(new_file_name)

            except Exception as e:
                failed_images.append(source_image_file_name)
                print(f"❌ Image failed: {source_image_file_name} ({e})")

        # ------------------------------------------------------------
        # Labels: pool-coordinate mapping -> save
        # ------------------------------------------------------------
        #
        # VALID pooling geometry:
        #   output ox,oy corresponds to input window:
        #       x in [ox*sh, ox*sh + kw - 1]
        #       y in [oy*sv, oy*sv + kh - 1]
        #
        # We map each (x,y) pixel to:
        #   ox = x // sh
        #   oy = y // sv
        # but only if that (x,y) lies inside the VALID-covered region:
        #   x <= (out_w-1)*sh + (kw-1)
        #   y <= (out_h-1)*sv + (kh-1)
        #
        for source_label_file_name in source_label_file_names:
            try:
                document = ImageAnnotationDocument.from_local_file(
                    subdirectory=source_label_subdir,
                    name=source_label_file_name,
                    extension=None,
                )

                # pooled output size using the same rule as Bitmap.pool_frame()
                W, H = int(document.width), int(document.height)
                out_w = (W - kw) // sh + 1
                out_h = (H - kh) // sv + 1

                if out_w <= 0 or out_h <= 0:
                    raise ValueError(
                        f"Pooling produced non-positive size: ({out_w},{out_h}) "
                        f"from ({W},{H}) kw,kh=({kw},{kh}) sh,sv=({sh},{sv})"
                    )

                # VALID-covered max input coordinate included by any window
                max_x_inclusive = (out_w - 1) * sh + (kw - 1)
                max_y_inclusive = (out_h - 1) * sv + (kh - 1)

                new_document = ImageAnnotationDocument(
                    name=document.name,
                    width=out_w,
                    height=out_h,
                )

                for label in document.data.labels:
                    old_bag = label.pixel_bag
                    new_bag = PixelBag()

                    for (x, y) in old_bag.pixels:
                        x = int(x)
                        y = int(y)

                        # If pixel lies outside the area any VALID window ever touches, drop it.
                        if x < 0 or y < 0 or x > max_x_inclusive or y > max_y_inclusive:
                            continue

                        ox = x // sh
                        oy = y // sv

                        if 0 <= ox < out_w and 0 <= oy < out_h:
                            new_bag.add(ox, oy)

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
        print(f"✅ SvetaPooler images: {len(out_image_files)} ok, {len(failed_images)} failed.")
        if failed_images:
            print("   Failed images:")
            for f in failed_images:
                print(f"   - {f}")

        print(f"✅ SvetaPooler labels: {len(out_label_files)} ok, {len(failed_labels)} failed.")
        if failed_labels:
            print("   Failed labels:")
            for f in failed_labels:
                print(f"   - {f}")

        return out_image_files, out_label_files
