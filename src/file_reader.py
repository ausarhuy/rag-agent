import json
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


class FileReader:
    @classmethod
    def load_pdf(cls, file):
        """Load a PDF file."""
        # Get the filename from file path
        file_name = os.path.basename(file)

        # Initialize Langchain's PyMuPDFLoader to load the PDF pages
        loader = PyMuPDFLoader(file)

        # Load the pages from the document
        pages = loader.load()
        total_pages = len(pages)
        documents = []

        # Loop through the pages
        for idx, page in enumerate(pages, start=1):
            # Append each page as Document object with modified metadata
            documents.append(
                Document(
                    page_content=page.page_content,
                    metadata=dict(
                        {
                            "file_name": file_name,
                            "page_no": str(idx),
                            "total_pages": str(total_pages),
                        }
                    ),
                )
            )

        return documents

    @classmethod
    def load_json(cls, file: str):
        """Load a JSON file."""
        with open(file, encoding="utf8") as f:
            json_data = json.load(f)

        return [Document(page_content=data["content"], metadata=data["metadata"]) for data in json_data]
