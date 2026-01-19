import os

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
