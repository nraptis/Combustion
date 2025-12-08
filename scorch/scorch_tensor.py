# scorch/scorch_tensor.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from image.bitmap import Bitmap
from image.rgba import RGBA

@dataclass
class ScorchTensor:
    """
    Core numeric tensor type for Scorch.

    Internal Rules:
    - ALWAYS normalized float32 in range [0, 1].
    - ALWAYS channel-first: (C, H, W).
    - C is 1 (grayscale) or 3 (RGB).
    """

    data: np.ndarray                  # MUST be float32 (C,H,W) normalized
    name: Optional[str] = None
    role: str = "generic"             # "image", "mask", etc.

    # ===============================================================
    # Bitmap → ScorchTensor
    # ===============================================================
    @classmethod
    def from_bitmap(
        cls,
        bmp: Bitmap,
        name: Optional[str] = None,
        role: str = "image",
        grayscale: bool = True,
    ) -> "ScorchTensor":
        """
        Convert a Bitmap to a ScorchTensor, ALWAYS normalized to [0,1].
        """
        if bmp.width <= 0 or bmp.height <= 0:
            return cls(np.zeros((0,), dtype=np.float32), name=name, role=role)

        # BGRA uint8 → (H, W, 4)
        bgra = bmp.export_opencv().astype(np.float32)

        # Extract channels
        b = bgra[:, :, 0]
        g = bgra[:, :, 1]
        r = bgra[:, :, 2]

        if grayscale:
            gray = RGBA.to_gray(r, g, b)     # H,W
            gray /= 255.0                                # normalize
            data = gray[np.newaxis, :, :]                # C=1,H,W
        else:
            r /= 255.0
            g /= 255.0
            b /= 255.0
            data = np.stack([r, g, b], axis=0)           # C=3,H,W

        return cls(data=data.astype(np.float32), name=name, role=role)
    
    @classmethod
    def from_bitmap_crop(
        cls,
        bmp: Bitmap,
        x: int,
        y: int,
        width: int,
        height: int,
        name: Optional[str] = None,
        role: str = "image",
        grayscale: bool = True,
    ) -> "ScorchTensor":
        """
        Crop region → Bitmap.crop → ScorchTensor.
        Always normalized.
        """
        cropped = bmp.crop(
            x=x,
            y=y,
            width=width,
            height=height)
        return cls.from_bitmap(cropped, name=name, role=role, grayscale=grayscale)
    

    # ===============================================================
    # Init hook
    # ===============================================================
    def __post_init__(self) -> None:
        # Force float32
        arr = np.asarray(self.data, dtype=np.float32)

        # Enforce normalization (final safety clamp)
        arr = np.clip(arr, 0.0, 1.0)

        # If 2D: promote to (1,H,W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]  # grayscale assumption

        # Validate shape
        if arr.ndim != 3:
            raise ValueError(f"ScorchTensor must be 3D (C,H,W), got shape {arr.shape}")

        self.data = arr

    # ===============================================================
    # Introspection
    # ===============================================================
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim
    

    def clone(self) -> "ScorchTensor":
        return ScorchTensor(self.data.copy(), name=self.name, role=self.role)

    def flatten(self) -> "ScorchTensor":
        c, h, w = self.data.shape
        return ScorchTensor(self.data.reshape(1, c * h * w), name=self.name, role=self.role)

    # ===============================================================
    # Framework Export
    # ===============================================================
    def to_numpy(self) -> np.ndarray:
        return self.data

    def to_torch(self):
        import torch
        return torch.from_numpy(self.data.copy())

    def to_tf(self):
        import tensorflow as tf
        arr = np.moveaxis(self.data, 0, -1)  # CHW → HWC
        return tf.convert_to_tensor(arr)

    # ===============================================================
    # ScorchTensor → Bitmap (for visualization)
    # ===============================================================
    def to_bitmap(self) -> Bitmap:
        """
        Convert this ScorchTensor back into a Bitmap.

        Assumes:
        - data ∈ [0,1]
        - shape = (C,H,W)
        - C = 1 or 3
        """
        arr = self.data

        if arr.size == 0:
            return Bitmap()

        c, h, w = arr.shape

        if c not in (1, 3):
            raise ValueError(f"Cannot convert tensor with C={c} to Bitmap")

        # Scale back to byte range
        img = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        # Allocate Bitmap
        bmp = Bitmap(w, h)

        if c == 1:
            # grayscale: broadcast to rgb
            gray = img[0, :, :]
            for x in range(w):
                col = bmp.rgba[x]
                for y in range(h):
                    v = int(gray[y, x])
                    px = col[y]
                    px.ri = v
                    px.gi = v
                    px.bi = v
                    px.ai = 255

        else:
            # RGB channels
            r = img[0, :, :]
            g = img[1, :, :]
            b = img[2, :, :]
            for x in range(w):
                col = bmp.rgba[x]
                for y in range(h):
                    px = col[y]
                    px.ri = int(r[y, x])
                    px.gi = int(g[y, x])
                    px.bi = int(b[y, x])
                    px.ai = 255

        return bmp
    
    def to_image(self):
        """
        Convert this ScorchTensor directly to a Pillow Image.

        Internally:
        - Uses to_bitmap() to rebuild a Bitmap
        - Then calls Bitmap.export_pillow() to get a Pillow Image
        """
        bmp = self.to_bitmap()
        return bmp.export_pillow()
