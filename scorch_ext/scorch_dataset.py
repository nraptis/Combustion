# scorch_ext/scorch_dataset.py

from __future__ import annotations
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

from scorch_ext.data_loader import DataLoader
from scorch_ext.tensor_load_helpers import iter_label_patches_from_pair
from scorch.scorch_tensor import ScorchTensor


class ScorchPatchClassificationDataset(Dataset):
    """
    Simple classification dataset:
      - walks your DataLoader (annotations + images)
      - uses image patches (from PixelBag frames) as inputs
      - uses label.name as the class
      - ignores masks for now (we're just doing classification)
    """

    def __init__(
        self,
        annotations_subdir: str,
        images_subdir: str | None = None,
        grayscale: bool = True,
    ) -> None:
        super().__init__()

        self.grayscale = grayscale
        self.class_to_index: Dict[str, int] = {}
        self.index_to_class: List[str] = []
        self.samples: List[Tuple[ScorchTensor, int]] = []

        loader = DataLoader(
            annotations_subdir=annotations_subdir,
            images_subdir=images_subdir,
        )

        for pair in loader:
            for img_tensor, _mask_tensor, label_name in iter_label_patches_from_pair(
                pair, grayscale=self.grayscale
            ):
                class_idx = self._get_or_add_class_index(label_name)
                self.samples.append((img_tensor, class_idx))

        print(f"[ScorchDataset] Loaded {len(self.samples)} samples "
              f"from {len(self.class_to_index)} classes.")

    # ----------------------------------------
    # Internal helpers
    # ----------------------------------------
    def _get_or_add_class_index(self, label_name: str) -> int:
        if label_name not in self.class_to_index:
            idx = len(self.index_to_class)
            self.class_to_index[label_name] = idx
            self.index_to_class.append(label_name)
        return self.class_to_index[label_name]

    # ----------------------------------------
    # Dataset interface
    # ----------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        scorch_tensor, class_idx = self.samples[idx]

        # Convert to PyTorch tensor (float32, C,H,W)
        x = scorch_tensor.to_torch()           # torch.float32, C,H,W
        y = torch.tensor(class_idx, dtype=torch.long)
        return x, y
