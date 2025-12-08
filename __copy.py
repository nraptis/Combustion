import os

def combine_python_files_recursive(root_folders, output_path):
    """
    Combine all .py files from multiple root folders (recursively)
    into one output text file.

    root_folders: list of folder paths
    output_path: path to write the combined file
    """

    # Normalize input: allow passing a single string
    if isinstance(root_folders, str):
        root_folders = [root_folders]

    with open(output_path, "w", encoding="utf-8") as outfile:

        for root in root_folders:
            root = os.path.abspath(root)

            outfile.write(f"# ========== ROOT: {root} ==========\n\n")

            # Walk the root recursively
            for folder, subfolders, files in os.walk(root):

                # Sort for stable output
                files = sorted(files)

                for filename in files:
                    if filename.endswith(".py"):
                        full_path = os.path.join(folder, filename)
                        rel_path = os.path.relpath(full_path, root)

                        # Header for each file
                        outfile.write(f"# ===== {rel_path} =====\n\n")

                        with open(full_path, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())

                        outfile.write("\n\n\n")  # spacing between files



combine_python_files_recursive(
    root_folders=[
        "scorch",
        "labels",
        
    ],
    output_path="fullcode.py.txt"
)