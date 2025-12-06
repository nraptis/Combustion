# scorch_tensor_test.py

from __future__ import annotations

from filesystem.file_io import FileIO
from image.bitmap import Bitmap
from scorch.scorch_tensor import ScorchTensor
from filesystem.file_utils import FileUtils

def aaa() -> None:
    # --------------------------------------------------
    # 1. Load the real test image via FileIO + Bitmap
    # --------------------------------------------------
    src_path = FileIO.local_file(
        subdirectory="testing",
        name="proto_cells_test_000",
        extension="png",
    )

    bmp = Bitmap.with_image(src_path)
    print(f"[scorch_tensor_test] Loaded bitmap: {bmp.width}x{bmp.height}")

    # --------------------------------------------------
    # 2. Full-image RGB tensor → back to Bitmap
    # --------------------------------------------------
    tensor_full_rgb = ScorchTensor.from_bitmap(
        bmp,
        name="proto_full_rgb",
        role="image",
        grayscale=False,   # RGB
    )
    print("[scorch_tensor_test] Full RGB tensor shape:", tensor_full_rgb.shape)

    bmp_full_rgb = tensor_full_rgb.to_bitmap()

    # this will create the directory.
    FileUtils.save_local_bitmap(bmp_full_rgb, "scorch_test_out", "from_tensor_uncropped_rgb", "png")
    
    print("[scorch_tensor_test] Saved..")

    # --------------------------------------------------
    # 3. Cropped RGB tensor (use a central patch)
    #    You can later swap these coords to true label bboxes.
    # --------------------------------------------------
    W, H = bmp.width, bmp.height
    crop_w = W // 2
    crop_h = H // 2
    crop_x = (W - crop_w) // 2
    crop_y = (H - crop_h) // 2

    tensor_crop_rgb = ScorchTensor.from_bitmap_crop(
        bmp,
        x=crop_x,
        y=crop_y,
        width=crop_w,
        height=crop_h,
        name="proto_crop_rgb",
        role="image",
        grayscale=False,   # RGB crop
    )
    print("[scorch_tensor_test] Cropped RGB tensor shape:", tensor_crop_rgb.shape)

    bmp_crop_rgb = tensor_crop_rgb.to_bitmap()

    FileUtils.save_local_bitmap(bmp_crop_rgb, "scorch_test_out", "from_tensor_crop_rgb", "png")
    
    print("[scorch_tensor_test] Saved...")

    # --------------------------------------------------
    # 4. Cropped GRAYSCALE tensor from the same region
    # --------------------------------------------------
    tensor_crop_gray = ScorchTensor.from_bitmap_crop(
        bmp,
        x=crop_x,
        y=crop_y,
        width=crop_w,
        height=crop_h,
        name="proto_crop_gray",
        role="image",
        grayscale=True,    # grayscale crop
    )
    print("[scorch_tensor_test] Cropped GRAY tensor shape:", tensor_crop_gray.shape)

    bmp_crop_gray = tensor_crop_gray.to_bitmap()
    FileUtils.save_local_bitmap(bmp_crop_gray, "scorch_test_out", "from_tensor_crop_gray", "png")
    print("[scorch_tensor_test] Saved...")


if __name__ == "__main__":
    aaa()
