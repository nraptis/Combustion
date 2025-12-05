# runner_scorch.py

from pathlib import Path
from scorch.annotation_loader import load_annotation_document
from filesystem.file_utils import FileUtils   # for image loading
from filesystem.file_io import FileIO


from image.bitmap import Bitmap
from image.rgba import RGBA

def debug_test_oob_crops():
    src_path = "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_000.png"
    bmp = Bitmap.with_image(src_path)

    red = RGBA(255, 0, 0, 255)

    W, H = bmp.width, bmp.height  # should be 256, 256
    crop_w = 256
    crop_h = 256

    # 1) Top-left: half off left + top
    tl = bmp.crop(
        x=-(crop_w // 4),   # -128
        y=-10,   # -128
        width=crop_w,
        height=crop_h,
        include_oob=True,
        oob_color=red,
    )
    tl.export_pillow().save("/Users/naraptis/Desktop/Combustion/testing/crop_tl_oobxx.png")

    # 2) Top-right: half off right + top
    tr = bmp.crop(
        x=W - (crop_w // 2),  # 256 - 128 = 128
        y=-(crop_h // 2),     # -128
        width=crop_w,
        height=crop_h,
        include_oob=True,
        oob_color=red,
    )
    tr.export_pillow().save("/Users/naraptis/Desktop/Combustion/testing/crop_tr_oob.png")

    # 3) Bottom-left: half off left + bottom
    bl = bmp.crop(
        x=-(crop_w // 2),     # -128
        y=H - (crop_h // 2),  # 256 - 128 = 128
        width=crop_w,
        height=crop_h,
        include_oob=True,
        oob_color=red,
    )
    bl.export_pillow().save("/Users/naraptis/Desktop/Combustion/testing/crop_bl_oob.png")

    # 4) Bottom-right: half off right + bottom
    br = bmp.crop(
        x=W - (crop_w // 4),  # 128
        y=H - (crop_h // 2),  # 128
        width=crop_w,
        height=crop_h,
        include_oob=True,
        oob_color=red,
    )
    br.export_pillow().save("/Users/naraptis/Desktop/Combustion/testing/crop_br_oob.png")

    # 5) Center OOB: bigger window centered on the image so it extends past all sides
    big_w = 66
    big_h = 66
    center_x = (W // 2) - (big_w // 2)  # 128 - 192 = -64
    center_y = (H // 2) - (big_h // 2)  # 128 - 192 = -64

    center = bmp.crop(
        x=center_x,
        y=center_y,
        width=big_w,
        height=big_h,
        include_oob=True,
        oob_color=red,
    )
    center.export_pillow().save("/Users/naraptis/Desktop/Combustion/testing/crop_center_oob.png")

def debug_test_overlap_crops():
    src_path = "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_000.png"
    bmp = Bitmap.with_image(src_path)

    W, H = bmp.width, bmp.height  # 256 x 256
    crop_w = 256
    crop_h = 256

    # Useful for naming
    outdir = "/Users/naraptis/Desktop/Combustion/testing/"

    # --------------------------------------------------
    # 1) Top-left — request goes negative in x and y
    # --------------------------------------------------
    tl = bmp.crop(
        x=-(crop_w // 2),   # -128
        y=-(crop_h // 2),   # -128
        width=crop_w,
        height=crop_h,
        include_oob=False,
    )
    tl.export_pillow().save(outdir + "crop_tl_overlap.png")
    print("TL size:", tl.width, tl.height)

    # --------------------------------------------------
    # 2) Top-right — goes OOB on top + right
    # --------------------------------------------------
    tr = bmp.crop(
        x=W - (crop_w // 2),   # 128
        y=-(crop_h // 2),      # -128
        width=crop_w,
        height=crop_h,
        include_oob=False,
    )
    tr.export_pillow().save(outdir + "crop_tr_overlap.png")
    print("TR size:", tr.width, tr.height)

    # --------------------------------------------------
    # 3) Bottom-left — OOB bottom + left
    # --------------------------------------------------
    bl = bmp.crop(
        x=-(crop_w // 2),      # -128
        y=H - (crop_h // 2),   # 128
        width=crop_w,
        height=crop_h,
        include_oob=False,
    )
    bl.export_pillow().save(outdir + "crop_bl_overlap.png")
    print("BL size:", bl.width, bl.height)

    # --------------------------------------------------
    # 4) Bottom-right — OOB bottom + right
    # --------------------------------------------------
    br = bmp.crop(
        x=W - (crop_w // 2),   # 128
        y=H - (crop_h // 2),   # 128
        width=crop_w,
        height=crop_h,
        include_oob=False,
    )
    br.export_pillow().save(outdir + "crop_br_overlap.png")
    print("BR size:", br.width, br.height)

    # --------------------------------------------------
    # 5) Center big crop (384×384) without OOB padding
    # --------------------------------------------------
    big_w = 88
    big_h = 44
    center_x = (W // 2) - (big_w // 2)   # -64
    center_y = (H // 2) - (big_h // 2)   # -64

    center = bmp.crop(
        x=center_x,
        y=center_y,
        width=big_w,
        height=big_h,
        include_oob=False,
    )
    center.export_pillow().save(outdir + "crop_center_overlap.png")
    print("Center size:", center.width, center.height)

def runner_scorch():
    print("=== SCORCH RUNNER START ===")


    print("\n=== SCORCH RUNNER DONE ===")

    debug_test_oob_crops()

    debug_test_overlap_crops()
