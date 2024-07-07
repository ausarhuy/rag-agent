from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document

from src.embeddings import embedding_function


class QueryCache:
    def __init__(self):
        # Initialize the embeddings model and cache vector store
        self.response_cache_store = Chroma(
            collection_name="caching",
            embedding_function=embedding_function,
        )

    def cache_query_response(self, query: str, response: str):
        # Create a Document object using query as the content and it's
        # response as metadata
        doc = Document(
            page_content=query,
            metadata={"response": response},
        )

        # Insert the Document object into cache vectorstore
        _ = self.response_cache_store.add_documents(documents=[doc])

    def find_similar_query_response(self, query: str, threshold: int):
        try:
            # Find similar query based on the input query
            sim_response = self.response_cache_store.similarity_search_with_score(
                query=query, k=1
            )

            # Return the response from the fetched entry if it's score is more than threshold
            return [
                {
                    "response": res[0].metadata["response"],
                }
                for res in sim_response
                if res[1] > threshold
            ]
        except Exception as e:
            raise Exception(e)
