import os
import streamlit as st
from langchain_core.documents import Document
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logger = get_logger('Langchain-Chatbot')

#decorator
def enable_chat_history(func):
    """setup chat history
    """
    if os.environ.get("OPENAI_API_KEY"):

        # clear chat history after switching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Which information from source documents do you need?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg: str, author: str):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_llm()->ChatOpenAI:
    """Log formatted question and answer.
    Returns:
        ChatOpenAI: LLM model client.
    """
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0, streaming=True)
    return llm

def print_qa(cls, question: str, answer: str):
    """Log formatted question and answer.
    """
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))

@st.cache_resource
def configure_embedding_model()->FastEmbedEmbeddings:
    """Get embedding model.
    Returns:
        FastEmbedEmbeddings: Embedding model.
    """
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v

def format_docs(docs: list[Document])->str:
    """Join the page content of the documents into a single string.
    Args:
        docs (list[str]): List of documents to join.
    Returns:
        str: The joined documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)