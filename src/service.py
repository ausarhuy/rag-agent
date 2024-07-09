"""
This function runs the backend. It starts the Langchain AI assistant: instanciate
all the Langchain chains for RAG and LLM.
"""

import streamlit as st
from langchain.chains import create_history_aware_retriever  # To create the retriever chain (predefined chain)
from langchain.chains import create_retrieval_chain  # To create the main chain (predefined chain)
from langchain.chains.combine_documents import create_stuff_documents_chain  # To create a predefined chain
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory, SQLChatMessageHistory, \
    StreamlitChatMessageHistory
from langchain_community.llms import CTransformers
from langchain_community.retrievers import BM25Retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyvi.ViTokenizer import tokenize

from src.config import *
from src.embeddings import embedding_function

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@st.cache_resource
def instanciate_ai_assistant_chain(model, temperature):
    """
    Instantiate retrievers and chains and return the main chain (AI Assistant).
    Steps: Retrieve and generate.
    """

    # Instanciate the model

    try:
        if model == "Vietnamese / Vistral":
            llm = CTransformers(model="models/llms/ggml-llms-7B-chat-q8.gguf", model_type="mistral",
                                config={'max_new_tokens': 1000, 'context_length': 4000, 'repetition_penalty': 1.1,
                                        "gpu_layers": 50, 'stream': True})
        elif model == "Google / Gemini 1.5":
            llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature, max_output_tokens=1000,
                                         stream=True)
        else:
            st.write("Error: No model available!")
            quit()

    except Exception as e:
        st.write("Error: Cannot instanciate any model!")
        st.write(f"Error: {e}")

    # Instanciate the ChromaDB

    try:
        vector_db = Chroma(embedding_function=embedding_function, collection_name=COLLECTION_NAME,
                           persist_directory="./chromadb")

        docs = vector_db.get()
        documents = docs["documents"]

    except Exception as e:
        st.write("Error: Cannot instanciate the vector database!")
        st.write(f"Error: {e}")

    # Instanciate the retrievers

    #vector_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": VECTORDB_MAX_RESULTS})
    fs = LocalFileStore("./docstore")
    docstore = create_kv_docstore(fs)

    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    vector_retriever = ParentDocumentRetriever(
        vectorstore=Chroma(embedding_function=embedding_function,
                           collection_name=COLLECTION_NAME,
                           persist_directory="./chromadb"),
        docstore=docstore, child_splitter=child_text_splitter)
    vector_retriever.search_kwargs["k"] = VECTORDB_MAX_RESULTS
    keyword_retriever = BM25Retriever.from_texts(documents, preprocess_func=tokenize)
    keyword_retriever.k = BM25_MAX_RESULTS

    ensemble_retriever = EnsembleRetriever(retrievers=[keyword_retriever, vector_retriever], weights=[0.4, 0.6])

    # Define the prompts

    contextualize_q_system_prompt = CONTEXTUALIZE_PROMPT

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "Câu hỏi: {input}")])

    qa_system_prompt = SYSTEM_PROMPT

    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "Câu hỏi: {input}")])

    # Instanciate the chains

    try:

        history_aware_retriever = create_history_aware_retriever(llm, ensemble_retriever, contextualize_q_prompt)

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        ai_assistant_chain = RunnableWithMessageHistory(rag_chain, get_session_history,
                                                        input_messages_key="input",
                                                        history_messages_key="chat_history",
                                                        output_messages_key="answer")

    except Exception as e:
        st.write("Error: Cannot instanciate the chains!")
        st.write(f"Error: {e}")
        ai_assistant_chain = None

    return ai_assistant_chain
