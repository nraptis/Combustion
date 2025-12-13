from __future__ import annotations

import random
from typing import List, Tuple, Optional

from image.bitmap import Bitmap


# -----------------------------
# Full-path pickers (yours)
# -----------------------------
def get_mask_path() -> str:
    masks = [
        "/Users/naraptis/Desktop/Combustion/images/ball_kernel_white_5_5.png",
        "/Users/naraptis/Desktop/Combustion/images/ball_kernel_red_5_5.png",
        "/Users/naraptis/Desktop/Combustion/images/soft_blur_kernel_3_3.png",
        "/Users/naraptis/Desktop/Combustion/images/identity_kernel_3_3.png",
        "/Users/naraptis/Desktop/Combustion/images/identity_kernel_5_3.png",
        "/Users/naraptis/Desktop/Combustion/images/identity_kernel_3_5.png",
        "/Users/naraptis/Desktop/Combustion/images/identity_kernel_5_5.png",
        "/Users/naraptis/Desktop/Combustion/images/mask_bad_shape.png",
        "/Users/naraptis/Desktop/Combustion/images/mask_muted_circle_13_13.jpg",
        "/Users/naraptis/Desktop/Combustion/images/mask_white_circle_13_13.png",
        "/Users/naraptis/Desktop/Combustion/images/mask_white_cross_5_5.png",

        # 15x15 neighbors
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_15_15/kernel_small_circle_neighbor_15_15_rr_uu.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_15_15/kernel_small_circle_neighbor_15_15_rr_dd.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_15_15/kernel_small_circle_neighbor_15_15_ll_uu.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_15_15/kernel_small_circle_neighbor_15_15_ll_dd.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_15_15/kernel_small_circle_neighbor_15_15_ll_d.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_15_15/kernel_small_circle_neighbor_15_15_ll_u.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_15_15/kernel_small_circle_neighbor_15_15_ll_c.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_15_15/kernel_small_circle_neighbor_15_15_rr_d.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_15_15/kernel_small_circle_neighbor_15_15_rr_u.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_15_15/kernel_small_circle_neighbor_15_15_rr_c.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_15_15/kernel_small_circle_neighbor_15_15_r_dd.png",

        # 11x11 neighbors
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_11_11/kernel_small_circle_neighbor_11_11_c_d.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_11_11/kernel_small_circle_neighbor_11_11_c_dd.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_11_11/kernel_small_circle_neighbor_11_11_c_u.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_11_11/kernel_small_circle_neighbor_11_11_c_uu.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_11_11/kernel_small_circle_neighbor_11_11_l_c.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_11_11/kernel_small_circle_neighbor_11_11_l_d.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_11_11/kernel_small_circle_neighbor_11_11_l_dd.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_11_11/kernel_small_circle_neighbor_11_11_l_u.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_11_11/kernel_small_circle_neighbor_11_11_l_uu.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_11_11/kernel_small_circle_neighbor_11_11_ll_c.png",
        "/Users/naraptis/Desktop/Combustion/images/kernel_small_circle_neighbor_11_11/kernel_small_circle_neighbor_11_11_ll_d.png",

        # Rectangular / stripe stress tests
        "/Users/naraptis/Desktop/Combustion/images/bad_egg_kernel_red_3_7.png",
        "/Users/naraptis/Desktop/Combustion/images/bad_egg_kernel_red_7_3.png",
        "/Users/naraptis/Desktop/Combustion/images/bad_egg_kernel_white_3_7.png",
        "/Users/naraptis/Desktop/Combustion/images/bad_egg_kernel_white_7_3.png",
        "/Users/naraptis/Desktop/Combustion/images/good_egg_kernel_white_5_9.png",
        "/Users/naraptis/Desktop/Combustion/images/good_egg_kernel_white_9_5.png",
        "/Users/naraptis/Desktop/Combustion/images/stripe_kernel_red_1_7.png",
        "/Users/naraptis/Desktop/Combustion/images/stripe_kernel_red_7_1.png",
        "/Users/naraptis/Desktop/Combustion/images/stripe_kernel_white_1_7.png",
        "/Users/naraptis/Desktop/Combustion/images/stripe_kernel_white_7_1.png",
    ]
    return random.choice(masks)


def get_image_path() -> str:
    images = [
        "/Users/naraptis/Desktop/Combustion/output_label_test/green_2_6_img.png",
        "/Users/naraptis/Desktop/Combustion/output_label_test/green_2_5_mask.png",
        "/Users/naraptis/Desktop/Combustion/output_label_test/green_2_5_img.png",

        "/Users/naraptis/Desktop/Combustion/input/Im013_1.jpg",
        "/Users/naraptis/Desktop/Combustion/input/Im012_1.jpg",
        "/Users/naraptis/Desktop/Combustion/input/Im011_1.jpg",
        "/Users/naraptis/Desktop/Combustion/input/Im010_1.jpg",
        "/Users/naraptis/Desktop/Combustion/input/Im009_1.jpg",
        "/Users/naraptis/Desktop/Combustion/input/Im008_1.jpg",
        "/Users/naraptis/Desktop/Combustion/input/Im007_1.jpg",
        "/Users/naraptis/Desktop/Combustion/input/Im006_1.jpg",

        "/Users/naraptis/Desktop/Combustion/output_label_test/red_1_2_mask.png",
        "/Users/naraptis/Desktop/Combustion/output_label_test/red_1_2_img.png",
        "/Users/naraptis/Desktop/Combustion/output_label_test/green_1_1_mask.png",
        "/Users/naraptis/Desktop/Combustion/output_label_test/green_1_1_img.png",
        "/Users/naraptis/Desktop/Combustion/output_label_test/blue_1_0_mask.png",
        "/Users/naraptis/Desktop/Combustion/output_label_test/blue_1_0_img.png",
        "/Users/naraptis/Desktop/Combustion/output_label_test/green_0_3_mask.png",
        "/Users/naraptis/Desktop/Combustion/output_label_test/green_0_3_img.png",

        "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_037.png",
        "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_038.png",
        "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_039.png",
        "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_040.png",
        "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_054.png",
        "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_064.png",
        "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_068.png",
        "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_092.png",
        "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_102.png",
        "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_116.png",
    ]
    return random.choice(images)


# -----------------------------
# Mask loading: bitmap -> mask[x][y] weights
# -----------------------------
def load_mask_weights_from_path(mask_path: str) -> List[List[float]]:
    """
    Loads mask image at mask_path into mask[x][y] float weights in [0,1]
    using the RED channel (keeps option for true red masks later).
    """
    mb = Bitmap.with_image(mask_path)
    if mb.width <= 0 or mb.height <= 0:
        raise ValueError(f"Mask bitmap has invalid size: {mb.width}x{mb.height} ({mask_path})")

    mask: List[List[float]] = [[0.0 for _y in range(mb.height)] for _x in range(mb.width)]
    for x in range(mb.width):
        col = mb.rgba[x]
        for y in range(mb.height):
            # red channel as weight, scaled to 0..1 (muted masks remain muted)
            mask[x][y] = float(col[y].ri) / 255.0
    return mask


def mask_dims(mask: List[List[float]]) -> Tuple[int, int]:
    w = len(mask)
    h = len(mask[0]) if w > 0 else 0
    return w, h


# -----------------------------
# Debug helpers
# -----------------------------
def first_diff(a: Bitmap, b: Bitmap):
    """Return (x,y,channel,av,bv,diff) for first mismatch, else None."""
    if a.width != b.width or a.height != b.height:
        return ("DIM", a.width, a.height, b.width, b.height)

    for x in range(a.width):
        ca = a.rgba[x]
        cb = b.rgba[x]
        for y in range(a.height):
            pa = ca[y]
            pb = cb[y]
            dr = abs(pa.ri - pb.ri)
            if dr:
                return (x, y, "r", pa.ri, pb.ri, dr)
            dg = abs(pa.gi - pb.gi)
            if dg:
                return (x, y, "g", pa.gi, pb.gi, dg)
            db = abs(pa.bi - pb.bi)
            if db:
                return (x, y, "b", pa.bi, pb.bi, db)
            da = abs(pa.ai - pb.ai)
            if da:
                return (x, y, "a", pa.ai, pb.ai, da)
    return None


def run_one(method_name: str, fn):
    try:
        out = fn()
        return True, out, None
    except Exception as e:
        return False, None, e


def random_params_mixed(img_w: int, img_h: int, mask_w: int, mask_h: int):
    """
    Returns (trim_h, trim_v, offset_x, offset_y, intended_valid).

    This version NEVER raises randrange/empty-range.
    If it can't find a valid (trim,offset) combo after N attempts,
    it falls back to an intentionally-invalid combo (so the test continues).
    """
    rx = mask_w // 2
    ry = mask_h // 2

    intended_valid = (random.random() < 0.80)  # 80% "try valid", 20% "force invalid"

    # -----------------------
    # Forced-invalid mode
    # -----------------------
    if not intended_valid:
        trim_h = random.randint(0, max(0, img_w // 2))
        trim_v = random.randint(0, max(0, img_h // 2))
        offset_x = random.choice([-(img_w + 10), img_w + 10, -(mask_w + 10), mask_w + 10])
        offset_y = random.choice([-(img_h + 10), img_h + 10, -(mask_h + 10), mask_h + 10])
        return trim_h, trim_v, offset_x, offset_y, False

    # -----------------------
    # Try-valid mode (re-roll)
    # -----------------------
    max_attempts = 200

    for _attempt in range(max_attempts):
        # trims: keep them reasonable so we actually compute meaningful outputs
        max_trim_h = max(0, (img_w - 1) // 2)
        max_trim_v = max(0, (img_h - 1) // 2)

        # cap trims so we don't constantly shrink to almost nothing
        trim_h = random.randint(0, min(max_trim_h, 64))
        trim_v = random.randint(0, min(max_trim_v, 64))

        start_x = trim_h
        end_x = img_w - trim_h  # exclusive
        start_y = trim_v
        end_y = img_h - trim_v  # exclusive

        out_w = end_x - start_x
        out_h = end_y - start_y
        if out_w <= 0 or out_h <= 0:
            continue  # no output, try again

        # Strict TRIM bounds for the whole output region (same logic as your preflight)
        # Need:
        #   min_x = start_x - rx + offset_x >= 0
        #   max_x = (end_x-1) + rx + offset_x < img_w
        # Solve for offset_x range:
        #   offset_x >= rx - start_x
        #   offset_x <= (img_w - 1) - ((end_x - 1) + rx)
        off_min_x = rx - start_x
        off_max_x = (img_w - 1) - ((end_x - 1) + rx)

        off_min_y = ry - start_y
        off_max_y = (img_h - 1) - ((end_y - 1) + ry)

        # If either range is empty, re-roll.
        if off_min_x > off_max_x or off_min_y > off_max_y:
            continue

        offset_x = random.randint(off_min_x, off_max_x)
        offset_y = random.randint(off_min_y, off_max_y)

        return trim_h, trim_v, offset_x, offset_y, True

    # -----------------------
    # Fallback: couldn't find valid params
    # -----------------------
    # This keeps the test moving and will count as a failure-match case.
    # (Also: if this happens a lot, it means your trims cap is too small for big kernels.)
    trim_h = random.randint(0, max(0, img_w // 2))
    trim_v = random.randint(0, max(0, img_h // 2))
    offset_x = random.choice([-(img_w + 10), img_w + 10, -(mask_w + 10), mask_w + 10])
    offset_y = random.choice([-(img_h + 10), img_h + 10, -(mask_h + 10), mask_h + 10])
    return trim_h, trim_v, offset_x, offset_y, False


def main():
    random.seed()

    trials = 1000
    tolerance = 1

    success_matches = 0
    failure_matches = 0
    parity_mismatches = 0
    output_mismatches = 0

    for i in range(1, trials + 1):
        mask_path = get_mask_path()
        img_path = get_image_path()

        # Force actual file IO each trial
        try:
            mask = load_mask_weights_from_path(mask_path)
            mw, mh = mask_dims(mask)
            bitmap = Bitmap.with_image(img_path)
        except Exception as e:
            print("\n[LOAD FAIL]")
            print(f"  trial={i}")
            print("  mask_path :", mask_path)
            print("  image_path:", img_path)
            print("  exc:", type(e).__name__, str(e))
            break

        trim_h, trim_v, offset_x, offset_y, intended_valid = random_params_mixed(
            bitmap.width, bitmap.height, mw, mh
        )

        native_ok, native_out, native_exc = run_one("native", lambda: bitmap.convolve(mask, trim_h, trim_v, offset_x, offset_y))
        opencv_ok, opencv_out, opencv_exc = run_one("opencv", lambda: bitmap.convolve_fast(mask, trim_h, trim_v, offset_x, offset_y, None))
        torch_ok,  torch_out,  torch_exc  = run_one("torch",  lambda: bitmap.convolve_torch(mask, trim_h, trim_v, offset_x, offset_y, device="cpu"))

        ok_tuple = (native_ok, opencv_ok, torch_ok)

        if len(set(ok_tuple)) != 1:
            parity_mismatches += 1
            print("\n[MISMATCH] success parity differs")
            print(f"  trial={i} intended_valid={intended_valid}")
            print("  mask_path :", mask_path)
            print("  image_path:", img_path)
            print(f"  img={bitmap.width}x{bitmap.height} mask={mw}x{mh}")
            print(f"  trim_h={trim_h} trim_v={trim_v} offset_x={offset_x} offset_y={offset_y}")
            print(f"  native_ok={native_ok} opencv_ok={opencv_ok} torch_ok={torch_ok}")
            if native_exc: print("  native_exc:", type(native_exc).__name__, str(native_exc))
            if opencv_exc: print("  opencv_exc:", type(opencv_exc).__name__, str(opencv_exc))
            if torch_exc:  print("  torch_exc :", type(torch_exc).__name__, str(torch_exc))
            break

        if not native_ok:
            # all failed: count as a failure-match if exception types match
            t0 = type(native_exc)
            if type(opencv_exc) is not t0 or type(torch_exc) is not t0:
                output_mismatches += 1
                print("\n[MISMATCH] all failed but exception types differ")
                print(f"  trial={i} intended_valid={intended_valid}")
                print("  mask_path :", mask_path)
                print("  image_path:", img_path)
                print(f"  img={bitmap.width}x{bitmap.height} mask={mw}x{mh}")
                print(f"  trim_h={trim_h} trim_v={trim_v} offset_x={offset_x} offset_y={offset_y}")
                print("  native_exc:", type(native_exc).__name__, str(native_exc))
                print("  opencv_exc:", type(opencv_exc).__name__, str(opencv_exc))
                print("  torch_exc :", type(torch_exc).__name__, str(torch_exc))
                break
            failure_matches += 1

        else:
            n_vs_o = native_out.compare(opencv_out, tolerance=tolerance)
            n_vs_t = native_out.compare(torch_out,  tolerance=tolerance)
            o_vs_t = opencv_out.compare(torch_out,  tolerance=tolerance)

            if not (n_vs_o and n_vs_t and o_vs_t):
                output_mismatches += 1
                print("\n[MISMATCH] outputs differ")
                print(f"  trial={i} intended_valid={intended_valid}")
                print("  mask_path :", mask_path)
                print("  image_path:", img_path)
                print(f"  img={bitmap.width}x{bitmap.height} mask={mw}x{mh}")
                print(f"  trim_h={trim_h} trim_v={trim_v} offset_x={offset_x} offset_y={offset_y}")
                print(f"  out sizes: native={native_out.width}x{native_out.height} opencv={opencv_out.width}x{opencv_out.height} torch={torch_out.width}x{torch_out.height}")
                print(f"  compares (tolerance={tolerance}): native-opencv={n_vs_o} native-torch={n_vs_t} opencv-torch={o_vs_t}")

                if not n_vs_o:
                    print("  first_diff native vs opencv:", first_diff(native_out, opencv_out))
                if not n_vs_t:
                    print("  first_diff native vs torch :", first_diff(native_out, torch_out))
                if not o_vs_t:
                    print("  first_diff opencv vs torch :", first_diff(opencv_out, torch_out))
                break

            success_matches += 1

        if i % 10 == 0:
            print(
                f"[OK] {i}/{trials} "
                f"success_matches={success_matches} "
                f"failure_matches={failure_matches} "
                f"parity_mismatches={parity_mismatches} "
                f"output_mismatches={output_mismatches}"
            )

    else:
        print(
            f"\n✅ All {trials} trials complete. "
            f"success_matches={success_matches}, failure_matches={failure_matches}, "
            f"parity_mismatches={parity_mismatches}, output_mismatches={output_mismatches} "
            f"(tolerance={tolerance})"
        )


if __name__ == "__main__":
    main()
