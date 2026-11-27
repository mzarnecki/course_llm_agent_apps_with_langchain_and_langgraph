import functools
import os

import streamlit as st
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit.logger import get_logger

logger = get_logger('Langchain-Chatbot')


def enable_chat_history(func):
    """Decorator: stop if the API key is missing, then show the chat history before func runs."""
    @functools.wraps(func)
    def execute(*args, **kwargs):
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Set OPENAI_API_KEY in the .env file or the environment.")
            st.stop()
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "Which information from source documents do you need?",
                }
            ]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])
        return func(*args, **kwargs)
    return execute


def display_msg(msg: str, author: str):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)


def configure_llm() -> ChatOpenAI:
    """Get the chat model client.
    Returns:
        ChatOpenAI: LLM model client.
    """
    return ChatOpenAI(model='gpt-6.1-terra', reasoning_effort='none', temperature=0)


def print_qa(cls, question: str, answer: str):
    """Log formatted question and answer.
    """
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))


@st.cache_resource
def configure_embedding_model() -> OpenAIEmbeddings:
    """Get embedding model.
    Returns:
        OpenAIEmbeddings: Embedding model.
    """
    return OpenAIEmbeddings(model="text-embedding-3-small")


def format_docs(docs: list[Document]) -> str:
    """Join the page content of the documents into a single string.
    Args:
        docs (list[str]): List of documents to join.
    Returns:
        str: The joined documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)
