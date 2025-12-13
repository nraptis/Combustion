# file_utils.py
from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from typing import Optional

from filesystem.file_io import FileIO
from image.bitmap import Bitmap
from PIL import Image

if TYPE_CHECKING:
    # Only imported for type checkers / IDEs, not at runtime
    from image.bitmap import Bitmap

class FileUtils:

    # ================================================================
    # TEXT UTILITIES
    # ================================================================

    @classmethod
    def load_text(cls, file_path: Path, encoding: str = "utf-8") -> str:
        data = FileIO.load(file_path)
        return data.decode(encoding)

    @classmethod
    def load_local_text(cls, subdirectory: Optional[str], name: str, extension: str, encoding="utf-8") -> str:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.load_text(path, encoding)

    @classmethod
    def save_text(cls, text: str, file_path: Path, encoding: str = "utf-8") -> Path:
        data = text.encode(encoding)
        return FileIO.save(data, file_path)

    @classmethod
    def save_local_text(cls, text: str, subdirectory: Optional[str], name: str, extension: str, encoding="utf-8") -> Path:
        data = text.encode(encoding)
        return FileIO.save_local(data, subdirectory, name, extension)

    # ================================================================
    # IMAGE UTILITIES
    # ================================================================

    @classmethod
    def load_image(cls, file_path: Path) -> Image.Image:
        path = Path(file_path).resolve()
        if path.is_file():
            image = Image.open(path)
            if image is not None:
                image.load()
                if image.width > 0 and image.height > 0:
                    return image
        base = path.with_suffix("")
        for extension in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".tiff"]:
            attempt_path = base.with_suffix(extension)
            if attempt_path.is_file():
                image = Image.open(attempt_path)
                if image is not None:
                    image.load()
                    if image.width > 0 and image.height > 0:
                        return image
        raise FileNotFoundError(f"Image not found: {path}")
    
    @classmethod
    def load_bitmap(cls, file_path: Path) -> Bitmap:
        image = cls.load_image(file_path)
        bitmap = Bitmap()
        bitmap.import_pillow(image)
        return bitmap

    @classmethod
    def load_local_image(cls, subdirectory: Optional[str], name: str, extension=None) -> Image.Image:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.load_image(path)
    
    @classmethod
    def load_local_bitmap(cls, subdirectory: Optional[str], name: str, extension=None) -> Bitmap:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.load_bitmap(path)

    @classmethod
    def save_image(cls, image: Image.Image, file_path: Path) -> Path:
        path = Path(file_path).resolve()
        FileIO._ensure_parent_dir(path)
        image.save(path)
        return path

    @classmethod
    def save_local_image(cls, image: Image.Image, subdirectory: Optional[str], name: str, extension="png") -> Path:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.save_image(image, path)
    
    @classmethod
    def save_local_bitmap(cls, bitmap: Optional[Bitmap], subdirectory: Optional[str], name: str, extension="png") -> Path:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.save_image(bitmap.export_pillow(), path)
