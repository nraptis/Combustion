# file_utils.py

from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from typing import Optional, Any

from filesystem.file_io import FileIO
from image.bitmap import Bitmap
from PIL import Image

import json

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
    def load_local_text(cls, subdirectory: Optional[str], name: str, extension: str, encoding: str = "utf-8") -> str:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.load_text(path, encoding)

    @classmethod
    def save_text(cls, text: str, file_path: Path, encoding: str = "utf-8") -> Path:
        data = text.encode(encoding)
        return FileIO.save(data, file_path)

    @classmethod
    def save_local_text(cls, text: str, subdirectory: Optional[str], name: str, extension: str, encoding: str = "utf-8") -> Path:
        data = text.encode(encoding)
        return FileIO.save_local(data, subdirectory, name, extension)

    # ================================================================
    # JSON UTILITIES
    # ================================================================

    @classmethod
    def load_json(cls, file_path: Path, encoding: str = "utf-8") -> Any:
        """
        Load JSON from an explicit path and return the parsed object (dict/list/etc).
        """
        text = cls.load_text(file_path, encoding=encoding)
        return json.loads(text)

    @classmethod
    def load_local_json(
        cls,
        subdirectory: Optional[str],
        name: str,
        extension: str = "json",
        encoding: str = "utf-8",
    ) -> Any:
        """
        Load JSON from a local (project-root) path.
        """
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.load_json(path, encoding=encoding)

    @classmethod
    def save_json(
        cls,
        obj: Any,
        file_path: Path,
        encoding: str = "utf-8",
        indent: int = 2,
        sort_keys: bool = False,
    ) -> Path:
        """
        Save a JSON-serializable object to an explicit path.
        """
        text = json.dumps(obj, indent=indent, sort_keys=sort_keys)
        return cls.save_text(text, file_path, encoding=encoding)

    @classmethod
    def save_local_json(
        cls,
        obj: Any,
        subdirectory: Optional[str],
        name: str,
        extension: str = "json",
        encoding: str = "utf-8",
        indent: int = 2,
        sort_keys: bool = False,
    ) -> Path:
        """
        Save a JSON-serializable object to a local (project-root) path.
        """
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.save_json(obj, path, encoding=encoding, indent=indent, sort_keys=sort_keys)

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
    def save_local_image(cls, image: Image.Image, subdirectory: Optional[str], name: str, extension: str = "png") -> Path:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.save_image(image, path)

    @classmethod
    def save_local_bitmap(cls, bitmap: Optional[Bitmap], subdirectory: Optional[str], name: str, extension: str = "png") -> Path:
        path = FileIO.local_file(subdirectory, name, extension)
        return cls.save_image(bitmap.export_pillow(), path)

    @classmethod
    def append_file_suffix(cls, file_name: str, suffix: str) -> str:
        """
        Append suffix before the extension (if any), preserving folders.

        Examples:
            "my_folder/my_file.txt", "_suffix" -> "my_folder/my_file_suffix.txt"
            "my_folder/my_file", "_suffix"     -> "my_folder/my_file_suffix"
            "my.file.txt", "_v2"               -> "my.file_v2.txt"
        """
        p = Path(file_name)
        parent = p.parent
        stem = p.stem
        ext = p.suffix
        new_name = f"{stem}{suffix}{ext}"
        if str(parent) == ".":
            return new_name
        return str(parent / new_name)