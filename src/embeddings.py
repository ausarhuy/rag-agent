from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from pyvi.ViTokenizer import tokenize

import sentence_transformers
from langchain_google_genai import GoogleGenerativeAIEmbeddings



class GeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str],
                        task_type: Optional[str] = None,
                        titles: Optional[List[str]] = None,
                        output_dimensionality: Optional[int] = None) -> List[List[float]]:
        embeddings = super().embed_documents(texts, task_type, titles, output_dimensionality)
        # Convert Repeated type to list type
        return list(map(list, embeddings))


class VietnameseEmbeddings(HuggingFaceEmbeddings):

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: tokenize*x.replace("\n", " ")), texts)
        if self.multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(
                texts, show_progress_bar=self.show_progress, **self.encode_kwargs
            )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
