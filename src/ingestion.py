from langchain_community.vectorstores.deeplake import DeepLake

from src.embeddings import GeminiEmbeddings
from src.utils import chunk_generator
from src.pdf_reader import PDFReader


class Ingestion:
    """Ingestion class for ingesting documents to vectorstore."""

    def __init__(self):
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.text_retriever = None
        self.embeddings = GeminiEmbeddings(
            model="models/embedding-001",
            task_type='retrieval_document',
        )
        self._initialize_text_vectorstore()

    def _initialize_text_vectorstore(self):
        # Initialize the vector store
        self.text_vectorstore = DeepLake(
            dataset_path="database/text_vectorstore",
            embedding=self.embeddings,
            overwrite=True,
            num_workers=4,
            verbose=False
        )

    def ingest_pdf(self, file: str):
        # Initialize the PDFReader and load the PDF as chunks
        loader = PDFReader()
        chunks = loader.load_pdf(file_path=file)

        print(f"Attempting to ingest {len(chunks)} embedding vectors from {file}")
        for chunk in chunk_generator(chunks):
            # Ingest the chunks
            _ = self.text_vectorstore.add_documents(chunk)

    def ingest_document(self, document: str):
        pass
