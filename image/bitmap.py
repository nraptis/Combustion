# bitmap.py

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List
import numpy as np
from PIL import Image
from image.rgba import RGBA
from filesystem.file_utils import FileUtils

# ----------------------------------------------------------------------
# Bitmap: rgba[x][y] with OpenCV + Pillow interop
# ----------------------------------------------------------------------

class Bitmap:
    """
    Bitmap with:
        - width, height
        - pixels stored as rgba[x][y] where:
            x = 0..width-1  (left to right)
            y = 0..height-1 (top to bottom)
    """

    def __init__(self, width: int = 0, height: int = 0) -> None:
        self.width: int = 0
        self.height: int = 0
        self.rgba: List[List[RGBA]] = []  # rgba[x][y]
        if width > 0 and height > 0:
            self.allocate(width, height)

    # --------------------------------------------------
    # The ONLY place we allocate the internal rgba array
    # --------------------------------------------------
    def allocate(self, width: int, height: int) -> None:
        """
        Resize the bitmap and allocate internal storage.
        This is the ONLY place rgba[][] is allocated.
        """
        self.width = int(width)
        self.height = int(height)

        # rgba[x][y]
        self.rgba = [
            [RGBA(0, 0, 0, 255) for _y in range(self.height)]
            for _x in range(self.width)
        ]

    # --------------------------------------------------
    # Expansion / copy
    # --------------------------------------------------
    def expand(self, width: int, height: int) -> None:
        """
        Expand this bitmap to at least (width, height).

        Existing pixels are preserved in the top-left corner.
        Newly exposed pixels are filled with opaque black (0,0,0,255).

        If the requested size is smaller than or equal to the current
        size in both dimensions, this is a no-op (no shrinking).
        """
        new_w = int(width)
        new_h = int(height)

        if new_w <= self.width and new_h <= self.height:
            # Nothing to do; we only expand, never shrink.
            return

        old_w = self.width
        old_h = self.height
        old_rgba = self.rgba

        # Allocate new storage
        new_rgba = [
            [RGBA(0, 0, 0, 255) for _y in range(new_h)]
            for _x in range(new_w)
        ]

        copy_w = min(old_w, new_w)
        copy_h = min(old_h, new_h)

        # Copy old pixels into the new buffer (top-left aligned)
        for x in range(copy_w):
            src_col = old_rgba[x]
            dst_col = new_rgba[x]
            for y in range(copy_h):
                src_px = src_col[y]
                dst_px = dst_col[y]
                dst_px.ri = src_px.ri
                dst_px.gi = src_px.gi
                dst_px.bi = src_px.bi
                dst_px.ai = src_px.ai

        # Swap in the new buffer
        self.width = new_w
        self.height = new_h
        self.rgba = new_rgba

    def copy(self) -> "Bitmap":
        """
        Deep copy this bitmap into a new Bitmap instance.
        Pixels are duplicated (no shared RGBA objects).
        """
        result = Bitmap()
        result.allocate(self.width, self.height)
        new_rgba = result.rgba

        for x in range(self.width):
            src_col = self.rgba[x]
            dst_col = new_rgba[x]
            for y in range(self.height):
                src_px = src_col[y]
                dst_px = dst_col[y]
                dst_px.ri = src_px.ri
                dst_px.gi = src_px.gi
                dst_px.bi = src_px.bi
                dst_px.ai = src_px.ai

        return result

    # --------------------------------------------------
    # Loading Methods: load via FileIO + import_pillow
    # --------------------------------------------------
    
    @classmethod
    def with_image(cls, file_path) -> "Bitmap":
        """
        Convenience constructor: create a Bitmap and load an image from
        an explicit file path via FileIO.load_image.
        """
        bmp = cls()
        bmp.load_image(file_path)
        return bmp
    
    @classmethod
    def with_local_image(
        cls,
        subdirectory: str | None = None,
        name: str | None = None,
        extension: str | None = None,
    ) -> "Bitmap":
        """
        Convenience constructor: create a Bitmap and load an image using
        FileIO.load_local_image (which uses FileIO.local for path building).
        """
        bmp = cls()
        bmp.load_local_image(
            subdirectory=subdirectory,
            name=name,
            extension=extension,
        )
        return bmp

    def load_image(self, file_path):
        """
        Create a Bitmap from an explicit file path using FileIO.load_image.
        """
        image = FileUtils.load_image(file_path)
        self.import_pillow(image)
        return self  # optional, enables chaining

    def load_local_image(
        self,
        subdirectory: str | None = None,
        name: str | None = None,
        extension: str | None = None,
    ) -> "Bitmap":
        """
        Create a Bitmap using FileIO.load_local_image
        (which uses FileIO.local for path building).
        """
        image = FileUtils.load_local_image(
            subdirectory=subdirectory,
            name=name,
            extension=extension,
        )
        self.import_pillow(image)
        return self  # optional, enables chaining

    # --------------------------------------------------
    # Import from OpenCV (NumPy array)
    # --------------------------------------------------
    def import_opencv(self, mat: np.ndarray) -> None:
        """
        Import from an OpenCV-style NumPy array.
        Supports:
            - H x W (grayscale)
            - H x W x 3 (BGR)
            - H x W x 4 (BGRA)
        """
        if mat is None:
            raise ValueError("mat is None")

        if mat.ndim == 2:
            # Grayscale: shape = (H, W)
            h, w = mat.shape
            self.allocate(w, h)
            for y in range(h):
                for x in range(w):
                    v = int(mat[y, x])
                    self.rgba[x][y] = RGBA(v, v, v, 255)

        elif mat.ndim == 3:
            h, w, c = mat.shape
            if c not in (3, 4):
                raise ValueError(f"Unsupported channel count: {c}")

            self.allocate(w, h)

            if c == 3:
                # BGR
                for y in range(h):
                    for x in range(w):
                        b, g, r = mat[y, x]
                        self.rgba[x][y] = RGBA(int(r), int(g), int(b), 255)
            elif c == 4:
                # BGRA
                for y in range(h):
                    for x in range(w):
                        b, g, r, a = mat[y, x]
                        self.rgba[x][y] = RGBA(int(r), int(g), int(b), int(a))

        else:
            raise ValueError(f"Unsupported mat.ndim = {mat.ndim}")

    # --------------------------------------------------
    # Import from Pillow Image
    # --------------------------------------------------
    def import_pillow(self, image: Image.Image) -> None:
        """
        Import from a Pillow Image.
        Converts to RGBA first to simplify handling.
        """
        if image is None:
            raise ValueError("image is None")

        img = image.convert("RGBA")
        w, h = img.size
        self.allocate(w, h)

        pixels = img.load()
        for x in range(w):
            for y in range(h):
                r, g, b, a = pixels[x, y]
                self.rgba[x][y] = RGBA(int(r), int(g), int(b), int(a))

    # --------------------------------------------------
    # Export to OpenCV (NumPy array)
    # --------------------------------------------------
    def export_opencv(self) -> np.ndarray:
        """
        Export to an OpenCV-style NumPy array (H x W x 4, BGRA).
        Caller can convert to BGR if desired:
            bgr = bgra[:, :, :3]
        """
        h = self.height
        w = self.width
        mat = np.zeros((h, w, 4), dtype=np.uint8)

        for x in range(w):
            for y in range(h):
                px = self.rgba[x][y]
                # OpenCV expects B, G, R, A
                mat[y, x, 0] = px.bi
                mat[y, x, 1] = px.gi
                mat[y, x, 2] = px.ri
                mat[y, x, 3] = px.ai

        return mat

    # --------------------------------------------------
    # Export to Pillow Image
    # --------------------------------------------------
    def export_pillow(self) -> Image.Image:
        """
        Export to a Pillow RGBA Image.
        """
        img = Image.new("RGBA", (self.width, self.height))
        pixels = img.load()

        for x in range(self.width):
            for y in range(self.height):
                px = self.rgba[x][y]
                pixels[x, y] = (px.ri, px.gi, px.bi, px.ai)

        return img

    # --------------------------------------------------
    # Flood fill: set every pixel to the same RGBA color
    # --------------------------------------------------
    def flood(self, color: RGBA) -> None:
        """
        Set every pixel in this bitmap to the given RGBA color.

        If the bitmap has zero width or height, this is a no-op.
        """
        if self.width <= 0 or self.height <= 0:
            return

        # Use the int components from the input color.
        r = color.ri
        g = color.gi
        b = color.bi
        a = color.ai

        for x in range(self.width):
            col = self.rgba[x]
            for y in range(self.height):
                px = col[y]
                px.ri = r
                px.gi = g
                px.bi = b
                px.ai = a

    # --------------------------------------------------
    # Internal helper: compute overlap for stamping
    # --------------------------------------------------
    def _compute_stamp_bounds(self, glyph: "Bitmap", x: int, y: int):
        """
        Compute the overlapping region between this bitmap (destination)
        and the glyph bitmap (source), given that the glyph's top-left
        should be placed at (x, y) in destination coordinates.

        Returns:
            (start_dx, end_dx, start_dy, end_dy, start_gx, start_gy)
        or None if there is no overlap.
        """
        gw, gh = glyph.width, glyph.height
        dw, dh = self.width, self.height
        if gw <= 0 or gh <= 0 or dw <= 0 or dh <= 0:
            return None
        start_dx = max(x, 0)
        start_dy = max(y, 0)
        end_dx = min(x + gw, dw)
        end_dy = min(y + gh, dh)
        if start_dx >= end_dx or start_dy >= end_dy:
            return None
        start_gx = start_dx - x
        start_gy = start_dy - y
        return (start_dx, end_dx, start_dy, end_dy, start_gx, start_gy)

    # --------------------------------------------------
    # Stamp: overwrite pixels from glyph into this bitmap
    # --------------------------------------------------
    def stamp(self, glyph: "Bitmap", x: int, y: int) -> None:
        """
        Stamp `glyph` onto this bitmap so that glyph (0,0)
        lands at destination (x,y).

        For now, we simply REPLACE the destination pixels with
        the glyph pixels (no alpha blending).

        All edge/off-grid cases are handled gracefully:
        - If the stamp is fully off-screen, nothing happens.
        - If the stamp is partially off-screen, only the visible
            part is drawn.
        """
        bounds = self._compute_stamp_bounds(glyph, x, y)
        if bounds is None:
            return
        start_dx, end_dx, start_dy, end_dy, start_gx, start_gy = bounds
        for dy in range(start_dy, end_dy):
            gy = start_gy + (dy - start_dy)
            for dx in range(start_dx, end_dx):
                gx = start_gx + (dx - start_dx)
                self.rgba[dx][dy] = glyph.rgba[gx][gy]

    # --------------------------------------------------
    # Stamp with classic alpha
    # --------------------------------------------------
    def stamp_alpha(self, glyph: "Bitmap", x: int, y: int) -> None:
        bounds = self._compute_stamp_bounds(glyph, x, y)
        if bounds is None:
            return
        start_dx, end_dx, start_dy, end_dy, start_gx, start_gy = bounds
        for dy in range(start_dy, end_dy):
            gy = start_gy + (dy - start_dy)
            for dx in range(start_dx, end_dx):
                gx = start_gx + (dx - start_dx)
                src_px = glyph.rgba[gx][gy]
                dst_px = self.rgba[dx][dy]
                self.rgba[dx][dy] = RGBA.blend_alpha(src_px, dst_px)
                
    # --------------------------------------------------
    # Stamp with additive blending
    # --------------------------------------------------
    def stamp_additive(self, glyph: "Bitmap", x: int, y: int) -> None:
        bounds = self._compute_stamp_bounds(glyph, x, y)
        if bounds is None:
            return
        start_dx, end_dx, start_dy, end_dy, start_gx, start_gy = bounds
        for dy in range(start_dy, end_dy):
            gy = start_gy + (dy - start_dy)
            for dx in range(start_dx, end_dx):
                gx = start_gx + (dx - start_dx)
                src_px = glyph.rgba[gx][gy]
                dst_px = self.rgba[dx][dy]
                self.rgba[dx][dy] = RGBA.blend_additive(src_px, dst_px)

    # --------------------------------------------------
    # Crop a sub-rectangle into a new Bitmap
    # --------------------------------------------------
    def crop(
        self,
        x: int,
        y: int,
        width: int,
        height: int) -> "Bitmap":
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0 or self.width <= 0 or self.height <= 0:
            return Bitmap()
        gw, gh = self.width, self.height
        dw, dh = width, height
        x_offset = -x
        y_offset = -y
        start_dx = max(x_offset, 0)
        start_dy = max(y_offset, 0)
        end_dx = min(x_offset + gw, dw)
        end_dy = min(y_offset + gh, dh)
        if start_dx >= end_dx or start_dy >= end_dy:
            return Bitmap()
        start_gx = start_dx - x_offset
        start_gy = start_dy - y_offset
        crop_w = end_dx - start_dx
        crop_h = end_dy - start_dy
        result = Bitmap(crop_w, crop_h)
        for dy in range(crop_h):
            sy = start_gy + dy
            for dx in range(crop_w):
                sx = start_gx + dx
                result.rgba[dx][dy] = self.rgba[sx][sy]
        return result
