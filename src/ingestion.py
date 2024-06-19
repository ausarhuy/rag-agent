from langchain_community.vectorstores.deeplake import DeepLake
from langchain_core.documents import Document

from src.embeddings import GeminiEmbeddings, VietnameseEmbeddings
from src.pdf_reader import PDFReader
from src.utils import chunk_generator


class Ingestion:
    """Ingestion class for ingesting documents to vectorstore."""

    def __init__(self, model_name="models/vietnamese-embedding"):
        self.text_vectorstore = None
        self.image_vectorstore = None
        self.text_retriever = None
        #self.embeddings = GeminiEmbeddings(model="models/text-multilingual-embedding-002")
        self.embeddings = VietnameseEmbeddings(model_name=model_name, model_kwargs={'device': 'cuda'})
        self._initialize_text_vectorstore()

    def _initialize_text_vectorstore(self):
        # Initialize the vector store
        self.text_vectorstore = Chroma(dataset_path="database/text_vectorstore", embedding=self.embeddings,
            overwrite=True, num_workers=4, verbose=False)

    def ingest_pdf(self, file: str):
        # Initialize the PDFReader and load the PDF as chunks
        loader = PDFReader()
        chunks = loader.load_pdf(file_path=file)

        print(f"Attempting to ingest {len(chunks)} embedding vectors from {file}")
        for chunk in chunk_generator(chunks):
            # Ingest the chunks
            _ = self.text_vectorstore.add_documents(chunk)

    def ingest_document(self, document: str):
        with open(document, "r") as f:
            data = json.load(f)

        chunks = []
        for chunk in data:
            chunk = Document(page_content=f"{chunk['title']}+\n{chunk['content']}",
                metadata=dict({"chapter": chunk["chapter"], "article": chunk["article"], "title": chunk["title"]}))
            chunks.append(chunk)
            # Ingest the chunks

        if self.model_name=="models/vietnamese-embeddings":
            _ = self.text_vectorstore.add_documents(chunks)
        else:
            # Gemini embeddings
            for chunk in chunk_generator(chunks):
                _ = self.text_vectorstore.add_documents(chunk)
