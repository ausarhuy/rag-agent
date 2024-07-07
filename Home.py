#!/usr/bin/env python
"""
This AI (Artificial Intelligence) assistant allows you to ask all kinds of questions regarding art
and the Belgian monarchy. To answer, the assistant queries the graphic databases.
Topology: backend = langchain (RAG + LLM), frontend = streamlit (chatbot + admin), assistant = main()
Start the app: streamlit run assistant.py
"""

import dotenv

dotenv.load_dotenv()


def main():
    """
    This is the main module: it will start the frontend (streamlit web interface) and
    backend (langchain AI assistant).
    """
    import streamlit as st

    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Streamlit! 👋")

    st.sidebar.success("Select a role.")

    st.markdown(
        """
        This AI (Artificial Intelligence) assistant allows you to ask all kinds of questions regarding art
        and the Belgian monarchy. To answer, the assistant queries the graphic databases.
        Topology: backend = langchain (RAG + LLM), frontend = streamlit (chatbot + admin), assistant = main()
        Start the app: streamlit run assistant.py
        """
    )


if __name__ == "__main__":
    main()
