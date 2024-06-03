import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter

import src.config as cfg


class PDFReader:
    """Custom PDF Loader to embed metadata with the pdfs."""

    def __init__(self):
        self.file_name = ""
        self.total_pages = 0

    def load_pdf(self, file_path):
        # Get the filename from file path
        self.file_name = os.path.basename(file_path)

        # Initialize Langchain's PyMuPDFLoader to load the PDF pages
        loader = PyMuPDFLoader(file_path)

        # Initialize the text splitter
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=cfg.PDF_CHARSPLITTER_CHUNKSIZE,
            chunk_overlap=cfg.PDF_CHARSPLITTER_CHUNK_OVERLAP,
        )

        # Load the pages from the document
        pages = loader.load()
        self.total_pages = len(pages)
        chunks = []

        # Loop through the pages
        for idx, page in enumerate(pages, start=1):
            # Append each page as Document object with modified metadata
            chunks.append(
                Document(
                    page_content=page.page_content,
                    metadata=dict(
                        {
                            "file_name": self.file_name,
                            "page_no": str(idx),
                            "total_pages": str(self.total_pages),
                        }
                    ),
                )
            )

        # Split the documents using splitter
        final_chunks = text_splitter.split_documents(chunks)
        return final_chunks
