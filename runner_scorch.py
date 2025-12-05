# runner_scorch.py

from pathlib import Path
from scorch.annotation_loader import load_annotation_document
from filesystem.file_utils import FileUtils   # for image loading
from filesystem.file_io import FileIO

def runner_scorch():
    print("=== SCORCH RUNNER START ===")

    # --------------------------------------------
    # Paths for your test files
    # --------------------------------------------
    ann_path = FileIO.local_file(name="////testing/proto_cells_test_000_annotations.json")
    img_path = "/Users/naraptis/Desktop/Combustion/testing/proto_cells_test_000.png"

    # --------------------------------------------
    # Load annotation document
    # --------------------------------------------
    print("\nLoading annotation JSON...")
    doc = load_annotation_document(ann_path)
    print("Loaded annotation document:")
    print(doc)
    

    # --------------------------------------------
    # Load PNG image
    # --------------------------------------------
    print("\nLoading corresponding PNG image...")
    img = FileUtils.load_image(Path(img_path))
    print(f"Loaded image: size={img.size}, mode={img.mode}")

    print("\n=== SCORCH RUNNER DONE ===")
