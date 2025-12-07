# data_loader_padded_test.py

from __future__ import annotations

from pathlib import Path

from scorch.data_loader import DataLoader as ScorchLoader
from scorch.tensor_load_helpers import (
    iter_label_patches_from_pair_fixed_size_centered,
)
from filesystem.file_utils import FileUtils


# Fixed slab size for all patches
FIXED_WIDTH = 128
FIXED_HEIGHT = 128


def main() -> None:
    print("=== data_loader_padded_test START ===")

    # Use the tiny training set
    loader = ScorchLoader(
        annotations_subdir="training_tiny",
        images_subdir="training_tiny",
    )

    for pair_index, pair in enumerate(loader):
        img_path = Path(pair.image_path)
        image_name = img_path.stem

        print(f"\n[PAIR {pair_index}] image={pair.image_path}")

        # Use the fixed-size centered iterator
        for patch_index, (img_tensor, mask_tensor, label_name) in enumerate(
            iter_label_patches_from_pair_fixed_size_centered(
                pair=pair,
                fixed_width=FIXED_WIDTH,
                fixed_height=FIXED_HEIGHT,
                grayscale=True,
            )
        ):
            c_i, h_i, w_i = img_tensor.shape
            c_m, h_m, w_m = mask_tensor.shape

            print(
                f"  [patch {patch_index}] "
                f"label={label_name!r} "
                f"img_shape=(C={c_i}, H={h_i}, W={w_i}) "
                f"mask_shape=(C={c_m}, H={h_m}, W={w_m})"
            )

            # Convert back to bitmaps
            bmp_img = img_tensor.to_bitmap()
            bmp_mask = mask_tensor.to_bitmap()

            # Save them for visual inspection
            piece_name = f"{image_name}_piece_{patch_index}"
            mask_name = f"{image_name}_mask_{patch_index}"

            FileUtils.save_local_bitmap(
                bmp_img,
                subdirectory="padded_tests",
                name=piece_name,
                extension="png",
            )

            FileUtils.save_local_bitmap(
                bmp_mask,
                subdirectory="padded_tests",
                name=mask_name,
                extension="png",
            )

            print(
                f"    -> saved fixed-size image as padded_tests/{piece_name}.png "
                f"and mask as padded_tests/{mask_name}.png"
            )

    print("\n=== data_loader_padded_test DONE ===")


if __name__ == "__main__":
    main()
