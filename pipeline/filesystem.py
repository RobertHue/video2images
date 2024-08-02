"""Filesystem related"""

# Python Module Index
import logging
import shutil


# 3rd-Party


def clear_directory(directory_path):
    """
    Clear all files and subdirectories from the specified directory.

    Args:
        directory_path (Path): Path object representing the directory to clear.

    Returns:
        None
    """
    if directory_path.exists():
        logging.info(f"Clearing output directory: {directory_path}")
        for item in directory_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        logging.info(
            f"Output directory does not exist. Creating new one: {directory_path}"
        )
