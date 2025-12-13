from image.bitmap import Bitmap
from filesystem.file_utils import FileUtils
from typing import List
import numpy as np
from typing import Optional

def load_mask_white_xy_255(subdirectory: Optional[str], name: str) -> List[List[int]]:
    try:
        bitmap = FileUtils.load_local_bitmap(subdirectory, name)
        mask_data = [[0 for _ in range(bitmap.height)] for _ in range(bitmap.width)]
        for x in range(bitmap.width):
            for y in range(bitmap.height):
                intensity = bitmap.rgba[x][y].ri
                mask_data[x][y] = intensity
        return mask_data
        print(mask_data)
    except FileNotFoundError:
        ...
        
    if subdirectory:
        raise FileNotFoundError(f"Mask file not found: {subdirectory}/{name}")
    else:
        raise FileNotFoundError(f"Mask file not found: {name}")
    
def load_mask_white_xy_weights(subdirectory: Optional[str], name: str) -> List[List[float]]:

    white_mask = load_mask_white_xy_255(subdirectory, name)

    width = len(white_mask)
    height = 0
    if width > 0:
        height = len(white_mask[0])
    else:
        return [[]]
    mask_data = [[0.0 for _ in range(height)] for _ in range(width)]
    _sum = 0
    maximum = 0
    for x in range(width):
        for y in range(height):
            _sum += white_mask[x][y]
            maximum = max(maximum, white_mask[x][y])
    if _sum > 0:
        scalar = float(maximum) / 255.0
        for x in range(width):
            for y in range(height):
                percent = float(white_mask[x][y]) / float(_sum)
                mask_data[x][y] = percent * scalar
    return mask_data
