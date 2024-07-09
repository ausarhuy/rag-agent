#!/usr/bin/env python

"""
This subpage runs the admin web interface.
"""
import os
import subprocess

import streamlit as st
from langchain.memory import ConversationBufferWindowMemory

from src.config import *
from src.ingestion import ingestion
from src.utils import delete_directory


def reset_conversation():
    """
    Reset the conversation: clear the chat history and clear the screen.
    """

    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.chat_history_k_window = ConversationBufferWindowMemory(k=MAX_MESSAGES_IN_MEMORY,
                                                                            return_messages=True)


def clear_memory_and_cache():
    st.cache_data.clear()
    st.cache_resource.clear()
    reset_conversation()


def delete_db():
    delete_directory("./chromadb")


def restart_db():
    command = ['bash', './db.sh', 'restart']
    st.write("Wait 20 seconds...")
    try:
        subprocess.run(command, capture_output=True, text=True, timeout=20)
    except Exception:
        st.write("")


def admin_frontend():
    st.set_page_config(page_title=ASSISTANT_NAME, page_icon=ASSISTANT_ICON)

    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL

    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = True

    if "input_password" not in st.session_state:
        st.session_state.input_password = ""

    st.title("Admin")

    st.sidebar.write(f"Model: {st.session_state.model} ({st.session_state.temperature})")

    # Ask admin password to access admin menu
    admin_password = os.getenv("ADMIN_PASSWORD", "YYYY")
    input_password = st.sidebar.text_input("Enter admin password: ", type="password",
                                           value=st.session_state.input_password)
    st.session_state.input_password = input_password
    if st.session_state.input_password != admin_password:
        st.session_state.authenticated = False
    else:
        st.session_state.authenticated = True

    if st.session_state.authenticated:

        # # # # # # # # # # # # # # # # # # # # #
        # Side bar window: second page (Admin)  #
        # # # # # # # # # # # # # # # # # # # # #

        options = ['Model and Temperature', 'Embed Pages in DB',
                   'Clear Memory and Streamlit Cache']
        choice = st.sidebar.radio("Make your choice: ", options)

        if choice == "Model and Temperature":
            st.caption("Change the model and the temperature for the present chat session.")
            model_list = [GEMINI_MENU, VISTRAL_MENU]
            st.session_state.model = st.selectbox('Model: ', model_list, DEFAULT_MENU_CHOICE)
            st.session_state.temperature = st.slider("Temperature: ", 0.0, 2.0, DEFAULT_TEMPERATURE)
            st.caption("OpenAI: 0-2, Anthropic: 0-1")

        elif choice == "Embed Pages in DB":
            # Embed data in Chroma DB
            # Load and index

            st.caption('Embed all data in the Chroma vector DB.')
            st.caption('Caution: Works only with files and DB running locally (server on which the app is running).')

            if st.button("Start Data Embed (locally only)"):
                ingestion.ingest_json(file="data/extracted_data.json")
                clear_memory_and_cache()
                st.write("Done!")

            if st.button("Delete DB (locally only)"):
                delete_db()
                restart_db()
                clear_memory_and_cache()
                st.write("Done!")

            if st.button("Restart DB (locally only)"):
                restart_db()
                st.write("Done!")

        elif choice == "Clear Memory and Streamlit Cache":
            st.caption("Clear the Langchain and Streamlit memory buffer and the Streamlit cache.")
            if st.button("Clear Memory and Streamlit Cache"):
                clear_memory_and_cache()
                st.write("Done!")


if __name__ == '__main__':
    admin_frontend()
