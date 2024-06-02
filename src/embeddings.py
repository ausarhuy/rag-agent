from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings


class GeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str],
                        task_type: Optional[str] = None,
                        titles: Optional[List[str]] = None,
                        output_dimensionality: Optional[int] = None) -> List[List[float]]:

        embeddings = super().embed_documents(texts, task_type, titles, output_dimensionality)
        # Convert Repeated type to list type
        return list(map(list, embeddings))
