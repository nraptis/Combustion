from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Import your new framer functions from the file you tweaked.
# If your framer file is named differently, change this import.
from cifar_training_video_frames import (
    load_reference_tile_128,
    render_training_frame,
    save_training_frame,
    INDEX_TO_POS,
)

class CifarVideoEmitter:
    """
    Bridges Conv2dWithFrames.emit(frame_tensor, meta) -> your frame-framer PNG sequence.

    - Expects frame_tensor: (out_ch, out_h, out_w) on CPU (your conv already .cpu()s it)
    - Uses only first 14 channels (because your layout has 14 dynamic slots)
    - Upscales each channel map to 128x128 with nearest-neighbor (blocky)
    """

    def __init__(
        self,
        reference_image_path: str = "reference_cat.png",
        output_directory: str = "training_video_frames",
        scale_factor_ref: int = 4,   # for 32->128 reference
    ):
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        self.reference_tile_128 = load_reference_tile_128(reference_image_path, scale=scale_factor_ref)

        self.frame_index = 0

        # Optional header fields (set from training loop)
        self.accuracy: Optional[float] = None
        self.epoch: Optional[int] = None
        self.epochs: Optional[int] = None
        self.progress: Optional[float] = None

    def set_header(
        self,
        accuracy: Optional[float] = None,
        epoch: Optional[int] = None,
        epochs: Optional[int] = None,
        progress: Optional[float] = None,
    ):
        self.accuracy = accuracy
        self.epoch = epoch
        self.epochs = epochs
        self.progress = progress

    @staticmethod
    def _feature_map_to_pil_128(f2d: torch.Tensor) -> Image.Image:
        """
        f2d: (H,W) float tensor (cpu)
        Returns: PIL RGB 128x128, blocky
        """
        if f2d.ndim != 2:
            raise ValueError("Expected 2D feature map")

        # Normalize per-map to [0,1] for visualization
        x = f2d.float()

        # stable visualization range (tweak the number, 1.0..5.0)
        CLAMP = 2.5
        x = x.clamp(-CLAMP, CLAMP)

        # map [-CLAMP, +CLAMP] -> [0, 1]
        x = (x + CLAMP) / (2 * CLAMP)

        # Shape to BCHW for interpolate
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        x = F.interpolate(x, size=(128, 128), mode="nearest")
        x = x.squeeze(0).squeeze(0)      # (128,128)

        # Convert grayscale -> RGB PIL
        arr = (x.numpy() * 255.0).clip(0, 255).astype(np.uint8)  # (128,128)
        rgb = np.stack([arr, arr, arr], axis=-1)                 # (128,128,3)
        return Image.fromarray(rgb).convert("RGB")

    def emit(self, frame_tensor: torch.Tensor, meta: Dict):
        """
        This matches Conv2dWithFrames.emit signature: emit(frame_tensor, meta_dict).
        frame_tensor: (out_ch, out_h, out_w) CPU
        """
        if frame_tensor.ndim != 3:
            raise ValueError(f"Expected (C,H,W), got {tuple(frame_tensor.shape)}")

        out_ch = frame_tensor.shape[0]

        # Build tiles dict (0..13). We fill them with channels 0..13.
        tiles: Dict[int, Image.Image] = {}

        for ch in range(24):
            tiles[ch] = self._feature_map_to_pil_128(frame_tensor[ch])

        """
        # Your layout has 14 slots
        for tile_idx in INDEX_TO_POS.keys():
            ch = tile_idx  # simple mapping: slot i shows channel i
            if ch < out_ch:
                tiles[tile_idx] = self._feature_map_to_pil_128(frame_tensor[ch])
            # else: leave empty -> red placeholder
        """

        img = render_training_frame(
            frame_index=self.frame_index,
            reference_tile_128=self.reference_tile_128,
            tiles=tiles,
            accuracy=self.accuracy,
            epoch=self.epoch,
            epochs=self.epochs,
            progress=self.progress,
        )
        save_training_frame(img, self.frame_index, output_dir=self.output_directory)

        self.frame_index += 1
