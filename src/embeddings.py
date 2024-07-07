from functools import lru_cache
import torch
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from pyvi.ViTokenizer import tokenize

import sentence_transformers

from src.config import EMBEDDING_MODEL


class VietnameseEmbeddings(HuggingFaceEmbeddings):

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(tokenize, texts))
        if self.multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(
                texts, show_progress_bar=self.show_progress, **self.encode_kwargs
            )

        return embeddings.tolist()


@lru_cache()
def get_embedding_model():
    return VietnameseEmbeddings(model_name=EMBEDDING_MODEL,
                                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})


embedding_function = get_embedding_model()
