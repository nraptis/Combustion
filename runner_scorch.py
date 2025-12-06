# runner_scorch.py

from filesystem.file_utils import FileUtils
from scorch.data_loader import DataLoader
from scorch.tensor_load_helpers import iter_label_patches_from_pair


def export_first_three_pairs_patches() -> None:
    """
    Use DataLoader, grab the first 3 AnnotationImagePairs,
    and for every label in each pair export:
      - image patch  -> output_label_test/<label>_<pair>_<idx>_img.png
      - mask patch   -> output_label_test/<label>_<pair>_<idx>_mask.png
    """
    loader = DataLoader(
        annotations_subdir="testing",
        images_subdir="testing",
    )

    print("going out")
    print("loader files = ", loader.annotation_files)

    for pair_index, pair in enumerate(loader):
        if pair_index >= 3:
            break

        print(f"[SCORCH] Processing pair {pair_index}: {pair.image_path}")

        for label_index, (img_tensor, mask_tensor, label_name) in enumerate(
            iter_label_patches_from_pair(pair, grayscale=True)
        ):
            # Image patch
            img_bmp = img_tensor.to_bitmap()
            FileUtils.save_local_bitmap(
                img_bmp,
                "output_label_test",
                f"{label_name.lower()}_{pair_index}_{label_index}_img.png",
            )

            # Mask patch
            mask_bmp = mask_tensor.to_bitmap()
            FileUtils.save_local_bitmap(
                mask_bmp,
                "output_label_test",
                f"{label_name.lower()}_{pair_index}_{label_index}_mask.png",
            )


def runner_scorch():
    print("=== SCORCH RUNNER START ===")
    export_first_three_pairs_patches()
    print("=== SCORCH RUNNER DONE ===")
