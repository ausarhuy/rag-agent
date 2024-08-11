"""
Ingestion class for ingesting documents to vectorstore.
"""

from typing import List

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import create_kv_docstore, LocalFileStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME, DOCSTORE_PATH, CHROMA_PATH
from src.embeddings import embedding_function
from src.file_reader import FileReader


class Ingestion:
    def __init__(self):
        self.text_vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=CHROMA_PATH,
        )
        self.child_text_splitter = RecursiveCharacterTextSplitter(separators="\n",
                                                                  chunk_size=CHUNK_SIZE,
                                                                  chunk_overlap=CHUNK_OVERLAP
                                                                  )
        self.docstore = create_kv_docstore(LocalFileStore(DOCSTORE_PATH))

    def _ingest_documents(self, docs: List[Document]):
        """Helper function to ingest a list of documents."""
        retriever = ParentDocumentRetriever(
            vectorstore=self.text_vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_text_splitter,
        )
        retriever.add_documents(docs)

    def ingest_pdf(self, file: str):
        """Ingest a PDF file."""
        docs = FileReader.load_pdf(file=file)
        self._ingest_documents(docs)

    def ingest_json(self, file: str):
        """Ingest a JSON file."""
        docs = FileReader.load_json(file=file)
        self._ingest_documents(docs)


ingestion = Ingestion()
