"""
This function runs the frontend web interface.
"""
import uuid

import streamlit as st
from langchain.memory import ConversationBufferWindowMemory

from src.config import *
from src.qa_service import instanciate_ai_assistant_chain
from src.utils import remove_chat_session


# Function defined in two files: should be moved in a module
def reset_conversation():
    """
    Reset the conversation: clear the chat history/session and clear the screen.
    """
    remove_chat_session(st.session_state.session_id)
    st.session_state.session_id = uuid.uuid4()
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.chat_history_k_window = ConversationBufferWindowMemory(k=MAX_MESSAGES_IN_MEMORY,
                                                                            return_messages=True)


def init_chat_session():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history_k_window = ConversationBufferWindowMemory(k=MAX_MESSAGES_IN_MEMORY,
                                                                                return_messages=True)  # Max k Q/A in the chat history for Langchain

    # Initialize chat session id
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4()

    # Initialize chat history (messages) for Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL

    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "input_password" not in st.session_state:
        st.session_state.input_password = ""


def assistant_frontend():
    """
    All related to Streamlit for the main page (about & chat windows) and connection with the Langchain backend.
    """

    st.set_page_config(page_title=ASSISTANT_NAME, page_icon=ASSISTANT_ICON)

    # Initialize chat history (chat_history) for LangChain
    init_chat_session()

    # Load, index, retrieve and generate

    ai_assistant_chain = instanciate_ai_assistant_chain(st.session_state.model, st.session_state.temperature)

    # # # # # # # #
    # Main window #
    # # # # # # # #

    st.image(LOGO_PATH, use_column_width=True)

    st.markdown(f"## {ASSISTANT_NAME}")
    st.caption("💬 A chatbot powered by Langchain and Streamlit")

    # # # # # # # # # # # # # #
    # Side bar window (About) #
    # # # # # # # # # # # # # #

    with st.sidebar:

        st.write(f"Model: {st.session_state.model} ({st.session_state.temperature})")
        st.write(ABOUT_TEXT)
        st.write(SIDEBAR_FOOTER)

    # # # # # # # # # # # #
    # Chat message window #
    # # # # # # # # # # # #

    with st.chat_message("assistant"):
        st.write(HELLO_MESSAGE)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if question := st.chat_input(USER_PROMPT):
        # Display user message in chat message container
        st.chat_message("user").markdown(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        try:

            # Call the main chain (AI assistant). invoke is replaced by stream to stream the answer.
            answer_container = st.empty()
            answer = ""
            for chunk in ai_assistant_chain.stream({"input": question, "chat_history": st.session_state.chat_history},
                                                   config={
                                                       "configurable": {"session_id": st.session_state.session_id}}):
                answer_chunk = str(chunk.get("answer"))
                if answer_chunk != "None":  # Because it writes NoneNone at the beginning
                    answer = answer + answer_chunk
                    answer_container.write(answer)

        except Exception as e:
            st.write("Error: Cannot invoke/stream the main chain!")
            st.write(f"Error: {e}")

        # Add Q/A to chat history for Langchain (chat_history)
        st.session_state.chat_history_k_window.save_context({"input": question}, {"output": answer})
        load_memory = st.session_state.chat_history_k_window.load_memory_variables({})
        st.session_state.chat_history = load_memory["history"]

        # Add Answer to chat history for Streamlit (messages)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Clear the conversation
        st.button(NEW_CHAT_MESSAGE, on_click=reset_conversation)


if __name__ == '__main__':
    assistant_frontend()
