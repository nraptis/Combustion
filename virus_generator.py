from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from filesystem.file_io import FileIO
from filesystem.file_utils import FileUtils

from image.bitmap import Bitmap
from image.rgba import RGBA

from labels.pixel_bag import PixelBag
from labels.data_label import DataLabel, DataLabelCollection
from labels.image_annotation_document import ImageAnnotationDocument


# ============================================================
# Output locations (subdirectories under your FileUtils local root)
# ============================================================

IMAGES_OUTPUT_DIR = "virus_synth_images"
ANNOTATIONS_OUTPUT_DIR = "virus_synth_annotations"


# ============================================================
# Dataset sizing
# ============================================================

OUTPUT_WIDTH = 8
OUTPUT_HEIGHT = 8

TRAIN_PREFIX = "train"
TEST_PREFIX = "test"

OUTPUT_COUNT_TRAIN = 10
OUTPUT_COUNT_TEST = 10


# ============================================================
# Background noise palette
# Each pixel is randomly chosen from these colors.
# (You can add more colors or swap alpha to taste.)
# ============================================================

BACKGROUND_COLORS: List[RGBA] = [
    RGBA(255, 0, 0, 255),    # red
    RGBA(0, 255, 0, 255),    # green
    RGBA(0, 0, 0, 255),      # black
]


# ============================================================
# Germ styling
# ============================================================

GERM_FILL_COLORS: List[RGBA] = [
    RGBA(255, 0, 0, 255),
]
GERM_STROKE_COLORS: List[RGBA] = [
    RGBA(0, 0, 0, 255),
]


# ============================================================
# Virus styling
# ============================================================

VIRUS_FILL_COLORS: List[RGBA] = [
    RGBA(0, 255, 0, 255),
]
VIRUS_STROKE_COLORS: List[RGBA] = [
    RGBA(0, 0, 0, 255),
]


# ============================================================
# Object counts
# ============================================================

GERM_COUNT_MIN = 1
GERM_COUNT_MAX = 1

VIRUS_COUNT_MIN = 1
VIRUS_COUNT_MAX = 1


# ============================================================
# Growth controls
#
# A germ/virus starts as 1 pixel, then expands in two phases:
#   Phase A: run SPREAD_A_STEPS times
#   Phase B: run SPREAD_B_STEPS times
#
# Each "step" picks N pixels from the frontier (4-neighbor ring).
# ============================================================

GERM_SPREAD_A_MIN = 0
GERM_SPREAD_A_MAX = 0
GERM_SPREAD_B_MIN = 0
GERM_SPREAD_B_MAX = 0

VIRUS_SPREAD_A_MIN = 0
VIRUS_SPREAD_A_MAX = 0
VIRUS_SPREAD_B_MIN = 0
VIRUS_SPREAD_B_MAX = 0

# How many frontier pixels to add per step (per object kind)
GERM_PICK_MIN = 1
GERM_PICK_MAX = 1

VIRUS_PICK_MIN = 1
VIRUS_PICK_MAX = 1


# ============================================================
# Placement attempts per object
# If an image can't place all objects, we discard that image and retry.
# ============================================================

PLACEMENT_ATTEMPT_COUNT = 2000

# Safety valve for "generate_many" so it never loops forever in silence.
# (Still: no exceptions, just stops if it hits this.)
MAX_IMAGE_ATTEMPTS = 200000


# ============================================================
# Types / helpers
# ============================================================

XY = Tuple[int, int]


@dataclass
class SynthResult:
    bmp: Bitmap
    doc: ImageAnnotationDocument


def _neighbors4(p: XY) -> List[XY]:
    x, y = p
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def _frontier4(shape: Set[XY]) -> List[XY]:
    """
    All 4-neighbors of any pixel in shape that are not already in shape.
    """
    out: Set[XY] = set()
    for p in shape:
        for q in _neighbors4(p):
            if q not in shape:
                out.add(q)
    return list(out)


def _build_shape_by_frontier(
    spread_a_steps: int,
    spread_b_steps: int,
    pick_min: int,
    pick_max: int,
) -> Set[XY]:
    """
    Build a shape in LOCAL coords (0,0 is the seed).
    No bounds/occupied checks here.
    """
    shape: Set[XY] = {(0, 0)}

    def grow_steps(steps: int) -> None:
        for _ in range(int(steps)):
            frontier = _frontier4(shape)
            if not frontier:
                return
            random.shuffle(frontier)
            k = random.randint(int(pick_min), int(pick_max))
            k = max(1, min(k, len(frontier)))
            for q in frontier[:k]:
                shape.add(q)

    grow_steps(spread_a_steps)
    grow_steps(spread_b_steps)
    return shape


def _stroke_from_frontier(shape: Set[XY]) -> Set[XY]:
    """
    Stroke pixels are exactly the frontier pixels.
    """
    return set(_frontier4(shape))


def _translate(shape: Set[XY], ox: int, oy: int) -> Set[XY]:
    ox = int(ox)
    oy = int(oy)
    return {(x + ox, y + oy) for (x, y) in shape}


def _fits_and_free(pixels: Set[XY], W: int, H: int, occupied: Set[XY]) -> bool:
    for (x, y) in pixels:
        if x < 0 or x >= W or y < 0 or y >= H:
            return False
        if (x, y) in occupied:
            return False
    return True


def _stamp_pixels(bmp: Bitmap, pixels: Set[XY], color: RGBA) -> None:
    """
    Dumb stamp: write ri/gi/bi/ai into each target pixel.
    """
    r, g, b, a = color.ri, color.gi, color.bi, color.ai
    for (x, y) in pixels:
        px = bmp.rgba[x][y]
        px.ri = r
        px.gi = g
        px.bi = b
        px.ai = a


def _save_annotation_json(subdir: str, stem: str, data: Dict) -> None:
    """
    Save as <stem>.json into subdir.
    Uses FileIO.local so it matches your filesystem conventions.
    """
    path = FileIO.local(subdirectory=subdir, name=stem, extension="json")
    # Ensure parent dir exists (FileIO likely does; but we do safe local mkdir)
    FileUtils.ensure_parent_dir(path) if hasattr(FileUtils, "ensure_parent_dir") else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ============================================================
# Placement
# ============================================================

def place_object(
    kind: str,
    bmp: Bitmap,
    W: int,
    H: int,
    occupied: Set[XY],
    labels: DataLabelCollection,
) -> bool:
    """
    Try PLACEMENT_ATTEMPT_COUNT times to place one object.
    Returns True on success, False on failure (no exception).
    """
    if kind == "germ":
        fill_color = random.choice(GERM_FILL_COLORS)
        stroke_color = random.choice(GERM_STROKE_COLORS)
        a_steps = random.randint(GERM_SPREAD_A_MIN, GERM_SPREAD_A_MAX)
        b_steps = random.randint(GERM_SPREAD_B_MIN, GERM_SPREAD_B_MAX)
        pick_min = GERM_PICK_MIN
        pick_max = GERM_PICK_MAX
        label_name = "germ"
    elif kind == "virus":
        fill_color = random.choice(VIRUS_FILL_COLORS)
        stroke_color = random.choice(VIRUS_STROKE_COLORS)
        a_steps = random.randint(VIRUS_SPREAD_A_MIN, VIRUS_SPREAD_A_MAX)
        b_steps = random.randint(VIRUS_SPREAD_B_MIN, VIRUS_SPREAD_B_MAX)
        pick_min = VIRUS_PICK_MIN
        pick_max = VIRUS_PICK_MAX
        label_name = "virus"
    else:
        return False

    for _attempt in range(int(PLACEMENT_ATTEMPT_COUNT)):
        local_fill = _build_shape_by_frontier(
            spread_a_steps=a_steps,
            spread_b_steps=b_steps,
            pick_min=pick_min,
            pick_max=pick_max,
        )
        local_stroke = _stroke_from_frontier(local_fill)

        # Reserve BOTH fill and stroke against occupied
        local_all = set(local_fill) | set(local_stroke)

        # Random origin
        x = random.randint(0, W - 1)
        y = random.randint(0, H - 1)

        fill = _translate(local_fill, x, y)
        stroke = _translate(local_stroke, x, y)
        all_px = fill | stroke

        if not _fits_and_free(all_px, W, H, occupied):
            continue

        # Commit occupancy
        occupied |= all_px

        # Stamp fill then stroke
        _stamp_pixels(bmp, fill, fill_color)
        _stamp_pixels(bmp, stroke, stroke_color)

        # Label stores the FULL object (fill+stroke)
        bag = PixelBag()
        for (px, py) in all_px:
            bag.add(px, py)
        labels.add_label(DataLabel(name=label_name, pixel_bag=bag))

        return True

    return False


# ============================================================
# One image
# ============================================================

def generate_one(name: str, W: int, H: int) -> Optional[SynthResult]:
    """
    Returns SynthResult or None if this attempt can't place all objects.
    """
    bmp = Bitmap(W, H)

    # Noisy background
    for x in range(W):
        col = bmp.rgba[x]
        for y in range(H):
            c = random.choice(BACKGROUND_COLORS)
            px = col[y]
            px.ri = c.ri
            px.gi = c.gi
            px.bi = c.bi
            px.ai = c.ai

    occupied: Set[XY] = set()
    labels = DataLabelCollection()

    germ_count = random.randint(GERM_COUNT_MIN, GERM_COUNT_MAX)
    virus_count = random.randint(VIRUS_COUNT_MIN, VIRUS_COUNT_MAX)

    # Place germs
    for _ in range(germ_count):
        if not place_object("germ", bmp, W, H, occupied, labels):
            return None

    # Place viruses
    for _ in range(virus_count):
        if not place_object("virus", bmp, W, H, occupied, labels):
            return None

    doc = ImageAnnotationDocument(name=name, width=W, height=H, data=labels)
    return SynthResult(bmp=bmp, doc=doc)


# ============================================================
# Many images
# ============================================================

def generate_many(prefix: str, count: int) -> None:
    """
    Writes exactly `count` images+annotations if possible.
    Never throws for placement failures; it just retries.
    """
    W = int(OUTPUT_WIDTH)
    H = int(OUTPUT_HEIGHT)

    made = 0
    attempts = 0

    while made < count and attempts < int(MAX_IMAGE_ATTEMPTS):
        attempts += 1
        stem = f"{prefix}_{made:04d}"

        res = generate_one(name=stem, W=W, H=H)
        if res is None:
            continue

        FileUtils.save_local_bitmap(res.bmp, IMAGES_OUTPUT_DIR, stem, "png")
        FileUtils.save_local_json(res.doc.to_json(), ANNOTATIONS_OUTPUT_DIR, stem)
        #_save_annotation_json(ANNOTATIONS_OUTPUT_DIR, stem, res.doc.to_json())


        made += 1
        if made == 1 or made % 5 == 0:
            print(f"[{prefix}] wrote {made}/{count} (attempts={attempts})")

    print(f"[{prefix}] done: wrote {made}/{count} (attempts={attempts})")


def main() -> None:
    # Deterministic runs if you want:
    # random.seed(911911)

    print("Generating train...")
    generate_many(TRAIN_PREFIX, int(OUTPUT_COUNT_TRAIN))

    print("Generating test...")
    generate_many(TEST_PREFIX, int(OUTPUT_COUNT_TEST))

    print("Done.")


if __name__ == "__main__":
    main()
