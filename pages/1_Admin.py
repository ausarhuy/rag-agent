"""
This subpage runs the admin web interface.
"""
import os
import subprocess
from pathlib import Path

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
    delete_directory(CHROMA_PATH)
    delete_directory(DOCSTORE_PATH)


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

        options = ['Model and Temperature', 'Embed Pages in DB', 'Clear Memory and Streamlit Cache']
        choice = st.sidebar.radio("Make your choice: ", options)

        if choice == "Model and Temperature":
            st.caption("Change the model and the temperature for the present chat session.")
            model_list = [GEMINI_MENU, VISTRAL_MENU]
            st.session_state.model = st.selectbox('Model: ', model_list, DEFAULT_MENU_CHOICE)
            st.session_state.temperature = st.slider("Temperature: ", 0.0, 2.0, DEFAULT_TEMPERATURE)

        elif choice == "Embed Pages in DB":

            st.caption('Embed all data in the Chroma vector DB.')

            with st.form(key="Upload document", clear_on_submit=True):
                submit = st.form_submit_button(label='Upload document')

                uploaded_file = st.file_uploader("Choose a file (JSON or PDF)", type=['json', 'pdf'])

                # this code block not working
                if submit and uploaded_file is not None:

                    save_path = Path(DATA_PATH, uploaded_file.name.split(".")[-1], uploaded_file.name)

                    with open(save_path, mode='wb') as w:
                        w.write(uploaded_file.getvalue())

                    if save_path.exists():
                        st.success(f'File {uploaded_file.name} is successfully saved!')
                clear_memory_and_cache()

            # List files in the specified directory
            files = [str(f) for f in Path(DATA_PATH).glob("**/*") if f.is_file()]

            if files:
                selected_file = st.selectbox("Select a file to embed:", files)

                if st.button("Start Embedding Document"):
                    try:
                        if selected_file.endswith(".json"):
                            ingestion.ingest_json(file=selected_file)
                        elif selected_file.endswith(".pdf"):
                            ingestion.ingest_pdf(file=selected_file)
                        clear_memory_and_cache()
                        st.success("Embed document successfully!")
                    except Exception as e:
                        st.error(f"Error: {e}")

            if st.button("Delete DB"):
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
