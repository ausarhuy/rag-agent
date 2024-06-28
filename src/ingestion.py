import json
import os

from langchain_community.vectorstores import DeepLake
from langchain_core.documents import Document

from src.embeddings import VietnameseEmbeddings
from src.pdf_reader import PDFReader


class Ingestion:
    """Ingestion class for ingesting documents to vectorstore."""

    def __init__(self, model_name="models/vietnamese-embedding", model_kwargs={'device': 'cpu'}):
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.text_retriever = None
        #self.embeddings = GeminiEmbeddings(model="models/text-multilingual-embedding-002")
        self.embeddings = VietnameseEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        self._initialize_text_vectorstore()

    def _initialize_text_vectorstore(self):
        # Initialize the vector store
        self.text_vectorstore = DeepLake(dataset_path="database/text_vectorstore", embedding=self.embeddings,
            overwrite=True, num_workers=4, verbose=False)

    def ingest_pdf(self, file: str):
        # Initialize the PDFReader and load the PDF as chunks
        loader = PDFReader()
        documents = loader.load_pdf(file_path=file)
        print(f"Attempting to ingest {len(documents)} embedding vectors from {file}")
        _ = self.text_vectorstore.add_documents(documents)

        # for chunk in chunk_generator(chunks):
        #     # Ingest the chunks
        #     _ = self.text_vectorstore.add_documents(chunk)

    def ingest_json(self, json_file: str):
        with open(json_file, "r") as f:
            data = json.load(f)

        # print(f"Attempting to ingest {len(documents)} embedding vectors from {json_file}")
        #
        # _ = self.text_vectorstore.add_documents(documents)
