from langchain.schema import Document
from langchain.vectorstores.deeplake import DeepLake
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import src.config as cfg


class CustomGPTCache:
    def __init__(self) -> None:
        # Initialize the embeddings model and cache vector store
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=cfg.GOOGLE_API_KEY,
            task_type="retrieval_document",
        )
        self.response_cache_store = DeepLake(
            dataset_path="database/cache_vectorstore",
            embedding=self.embeddings,
            read_only=False,
            num_workers=4,
            verbose=False,
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
