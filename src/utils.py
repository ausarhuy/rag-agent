from itertools import islice


def chunk_generator(chunks, size=100):
    """
    Generates smaller chunk of the given chunks.

    Args:
        chunks (list): The input list.
        chunk_size (int, optional): Size of each chunk. Defaults to 100.

    Yields:
        generator: A chunk of the input chunks.
    """
    it = iter(chunks)
    return iter(lambda: tuple(islice(it, size)), ())
