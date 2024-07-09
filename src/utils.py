"""
Miscellaneous functions, including function to chunk and embed files.
"""
import shutil
from itertools import islice


def delete_directory(dir_path):
    try:
        shutil.rmtree(dir_path)
        print(f"Directory '{dir_path}' and all its contents have been deleted successfully")
    except FileNotFoundError:
        print(f"Error: Directory '{dir_path}' does not exist")
    except PermissionError:
        print(f"Error: Permission denied to delete '{dir_path}'")
    except Exception as e:
        print(f"Error: {e}")


def chunk_generator(chunks, size=100):
    """
    Generates smaller chunk of the given chunks.

    Args:
        chunks (list): The input list.
        size (int, optional): Size of each chunk. Defaults to 100.

    Yields:
        generator: A chunk of the input chunks.
    """
    it = iter(chunks)
    return iter(lambda: tuple(islice(it, size)), ())
