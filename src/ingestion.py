from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.stores import InMemoryByteStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
from src.embeddings import embedding_function
from src.pdf_reader import PDFReader


class Ingestion:
    """Ingestion class for ingesting documents to vectorstore."""

    def __init__(self):
        self.text_vectorstore = Chroma(collection_name=COLLECTION_NAME,
                                       embedding_function=embedding_function,
                                       persist_directory="./chromadb")
        self.text_retriever = None

    def ingest_pdf(self, file: str):
        # Initialize the PDFReader and load the PDF as chunks
        loader = PDFReader()
        documents = loader.load_pdf(file_path=file)
        print(f"Attempting to ingest {len(documents)} embedding vectors from {file}")
        _ = self.text_vectorstore.add_documents(documents)

    def ingest_documents(self, data: str):
        loader = DirectoryLoader(data, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
        docs = loader.load()

        store = InMemoryByteStore()

        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        retriever = ParentDocumentRetriever(
            vectorstore=self.text_vectorstore,
            docstore=store,
            child_splitter=child_text_splitter,
        )

        retriever.add_documents(docs)
