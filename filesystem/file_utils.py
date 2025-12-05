# file_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from filesystem.file_io import FileIO
from PIL import Image

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
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)
        img.load()
        return img

    @classmethod
    def load_local_image(cls, subdirectory: Optional[str], name: str, extension="png") -> Image.Image:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.load_image(path)

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
