import json
from typing import List

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import create_kv_docstore, LocalFileStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME, DOCSTORE_PATH
from src.embeddings import embedding_function
from src.pdf_reader import PDFReader


class Ingestion:
    """Ingestion class for ingesting documents to vectorstore."""

    def __init__(self):
        self.text_vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory="./chromadb",
        )
        self.child_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
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
        loader = PDFReader()
        docs = loader.load_pdf(file_path=file)
        print(f"Attempting to ingest {len(docs)} embedding vectors from {file}")
        self._ingest_documents(docs)

    def ingest_json(self, file: str):
        """Ingest a JSON file."""
        with open(file, encoding="utf8") as f:
            json_data = json.load(f)

        docs = [Document(page_content=data["content"], metadata=data["metadata"]) for data in json_data]
        self._ingest_documents(docs)


ingestion = Ingestion()
