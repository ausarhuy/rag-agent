from langchain.vectorstores.deeplake import DeepLake
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
)

from src.pdf_reader import PDFReader


class Ingestion:
    """Ingestion class for ingesting documents to vectorstore."""

    def __init__(self):
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.text_retriever = None
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type='retrieval_document',
        )

    def ingest_documents(
            self,
            file: str,
    ):
        # Initialize the PDFReader and load the PDF as chunks
        loader = PDFReader()
        chunks = loader.load_pdf(file_path=file)

        # Initialize the vector store
        vstore = DeepLake(
            dataset_path="database/text_vectorstore",
            embedding=self.embeddings,
            overwrite=True,
            num_workers=4,
            verbose=False,
        )

        # Ingest the chunks
        _ = vstore.add_documents(chunks)
