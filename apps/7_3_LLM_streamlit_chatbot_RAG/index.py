import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from lib import utils
from lib.chat_prompt import ChatPrompt
from lib.document_loader import DocumentLoader

load_dotenv()

st.set_page_config(page_title="Chat", page_icon="📄")
st.header('AI chat - search for information in source documents with RAG')
st.write(
    "This application has access to custom documents and can respond to user "
    "queries by referring to the content within those documents."
)


@st.cache_resource(show_spinner="Indexing source documents...")
def build_retriever():
    """Embed the source documents once per server process, not on every Streamlit rerun."""
    chunks = DocumentLoader().load_chunks()
    vectordb = InMemoryVectorStore.from_documents(chunks, utils.configure_embedding_model())
    # k=2 misses the dated chunks needed to tell apart people sharing a name
    return vectordb.as_retriever(search_kwargs={'k': 4})


class CustomDocChatbot:

    def __init__(self):
        self.llm = utils.configure_llm()
        self.retriever = build_retriever()
        self.chain = ChatPrompt().get_prompt() | self.llm | StrOutputParser()

    @staticmethod
    def chat_history() -> list:
        """Earlier turns from the Streamlit session, without the current question."""
        roles = {"user": HumanMessage, "assistant": AIMessage}
        return [roles[m["role"]](content=m["content"]) for m in st.session_state.messages[:-1]]

    @utils.enable_chat_history
    def main(self):
        user_query = st.chat_input(placeholder="Ask for information from documents")
        if not user_query:
            return

        utils.display_msg(user_query, 'user')

        with st.chat_message("assistant"):
            with st.spinner('Searching documents...'):
                context_docs = self.retriever.invoke(user_query)
            response_text = st.write_stream(self.chain.stream({
                "input": user_query,
                "context": utils.format_docs(context_docs),
                "chat_history": self.chat_history(),
            }))
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            utils.print_qa(CustomDocChatbot, user_query, response_text)

            for doc in context_docs:
                filename = os.path.basename(doc.metadata['source'])
                with st.popover(f":blue[Source document: {filename}]"):
                    st.caption(doc.page_content)


if __name__ == "__main__":
    CustomDocChatbot().main()
