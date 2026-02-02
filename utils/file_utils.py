import os
from pathlib import Path 
import shutil
import re

def list_files_in_directory(directory):
    """
    Scans through the given directory and returns a list of all files.

    Parameters:
        directory (str): The path to the directory to scan.

    Returns:
        list: A list of file paths in the directory.
    """
    files = []
    for entry in os.scandir(directory):
        if entry.is_file():  # Check if the entry is a file
            files.append(entry.path)
    return files

def return_highest_eval_seed_directory(parent_dir):
    parent = Path(parent_dir)

    # 1. Check existence
    if not parent.exists() or not parent.is_dir():
        print(f"Directory does not exist: {parent}")
        return None

    pattern = re.compile(r"eval_seed_(\d+)$")

    matches = []

    # 2. Scan subdirectories
    for p in parent.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                seed = int(m.group(1))
                matches.append((seed, p))

    if not matches:
        print("No eval_seed_* directories found.")
        return None

    # 3. Sort by seed number
    matches.sort(key=lambda x: x[0])

    # 4. Pick highest
    highest_seed, highest_path = matches[-1]

    return highest_seed, highest_path
