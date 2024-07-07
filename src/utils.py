"""
Miscellaneous functions, including function to chunk and embed files.
"""
import shutil
from itertools import islice

from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.storage import SQLStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
from src.embeddings import embedding_function


def load_documents(path: str) -> None:
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    docs = loader.load()

    store = SQLStore()

    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    retriever = ParentDocumentRetriever(
        vectorstore=Chroma(embedding_function=embedding_function, collection_name=COLLECTION_NAME,
                           persist_directory="./chromadb"), docstore=store, child_splitter=child_text_splitter)

    retriever.add_documents(docs)


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
