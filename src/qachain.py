from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.deeplake import DeepLake
from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

import src.config as cfg
from src.cache import CustomGPTCache
from src.embeddings import GeminiEmbeddings


class QAChain:
    def __init__(self):
        # Initialize Gemini Embeddings
        self.embeddings = GeminiEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_query"
        )

        # Initialize Gemini Chat model
        self.model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3
        )

        # Initialize GPT Cache
        self.cache = CustomGPTCache()
        self.text_vectorstore = None
        self.text_retriever = None

    def ask_question(self, query):
        try:
            # Search for similar query response in cache
            cached_response = self.cache.find_similar_query_response(
                query=query, threshold=cfg.CACHE_THRESHOLD
            )

            # If similar query response is present, return it
            if len(cached_response) > 0:
                print("Using cache")
                result = cached_response[0]["response"]
            # Else generate response for the query
            else:
                print("Generating response")
                result = self.generate_response(query=query)
        except Exception as _:
            print("Exception raised. Generating response.")
            result = self.generate_response(query=query)

        return result

    def generate_response(self, query: str):
        # Initialize the vectorstore and retriever object
        vstore = DeepLake(
            dataset_path="database/text_vectorstore",
            embedding=self.embeddings,
            read_only=True,
            num_workers=4,
            verbose=False,
        )
        retriever = vstore.as_retriever(search_type="similarity")
        retriever.search_kwargs["distance_metric"] = "cos"
        retriever.search_kwargs["fetch_k"] = 20
        retriever.search_kwargs["k"] = 15

        # Write prompt to guide the LLM to generate response
        prompt_template = """
        <YOUR PROMPT HERE>
        Context: {context}
        Question: {question}

        Answer:
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}

        # Create Retrieval QA chain
        qa = RetrievalQA.from_chain_type(
            llm=self.model,
            retriever=retriever,
            verbose=False,
            chain_type_kwargs=chain_type_kwargs
        )

        # Run the QA chain and store the response in cache
        result = qa.invoke({"query": query})["result"]
        self.cache.cache_query_response(query=query, response=result)

        return result
