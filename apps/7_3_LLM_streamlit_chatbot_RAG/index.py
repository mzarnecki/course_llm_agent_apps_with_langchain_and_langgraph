import os
import streamlit as st

from lib import utils
from lib.ChatPrompt import ChatPrompt
from lib.DocumentLoader import DocumentLoader

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

st.set_page_config(page_title="Chat", page_icon="📄")
st.header('AI chat - search for information in source documents with RAG')
st.write(
    'This application has access to custom documents and can respond to user queries by referring to the content within those documents.')


class CustomDocChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()
        self.document_loader = DocumentLoader()
        # Store for session history
        if 'langchain_store' not in st.session_state:
            st.session_state.langchain_store = {}
        self.store = st.session_state.langchain_store
        self.chat_prompt = ChatPrompt().get_prompt()
        # import source documents
        chunks = self.document_loader.load_chunks()
        vectordb = DocArrayInMemorySearch.from_documents(chunks, self.embedding_model)
        self.retriever = vectordb.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 2, 'fetch_k': 4}
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retrieves the chat message history for a specific session
        Args:
            session_id (str): The session ID
        Returns:
            BaseChatMessageHistory: The chat message history for the session
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
            # Initialize with existing messages from streamlit session
            if "messages" in st.session_state:
                for msg in st.session_state["messages"]:
                    if msg["role"] == "user":
                        self.store[session_id].add_message(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        self.store[session_id].add_message(AIMessage(content=msg["content"]))
        return self.store[session_id]

    def prepare_rag_chain(self)->RunnableWithMessageHistory:
        """Prepares the RAG chain

        Returns:
            RunnableWithMessageHistory: The RAG chain
        """
        # Retrieval chain - direct retrieval without contextualization
        retrieval_chain = RunnablePassthrough.assign(
            context=lambda x: self.retriever.invoke(x["input"])
        ) | RunnablePassthrough.assign(
            formatted_context=lambda x: utils.format_docs(x["context"])
        )

        # Complete RAG chain using LCEL
        rag_chain = (
            retrieval_chain
            | RunnablePassthrough.assign(answer=(
                {
                    "context": itemgetter("formatted_context"),
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history")
                }
                | self.chat_prompt
                | self.llm
                | StrOutputParser())
            )
        )

        # Wrap with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return conversational_rag_chain


    @utils.enable_chat_history
    def main(self):
        user_query = st.chat_input(placeholder="Ask for information from documents")

        if user_query:
            # Get or initialize the session history before building the chain
            session_history = self.get_session_history("default_session")

            rag_chain = self.prepare_rag_chain()

            utils.display_msg(user_query, 'user')

            # Add user message to LangChain history
            session_history.add_message(HumanMessage(content=user_query))

            with st.spinner('Preparing response...'):
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    response_text = ""
                    context_docs = None

                    # Stream the response chunks
                    for chunk in rag_chain.stream(
                        {"input": user_query},
                        config={"configurable": {"session_id": "default_session"}}
                    ):
                        # Capture the answer chunks for streaming
                        if "answer" in chunk:
                            # For streaming, answer will come in parts
                            if isinstance(chunk["answer"], str):
                                response_text += chunk["answer"]
                                response_placeholder.markdown(response_text)
                        # Capture context documents for references
                        if "context" in chunk:
                            context_docs = chunk["context"]

                    # Store the complete response in both places
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    # Add assistant message to LangChain history
                    session_history.add_message(AIMessage(content=response_text))

                    utils.print_qa(CustomDocChatbot, user_query, response_text)

                    # Show references if available
                    if context_docs:
                        for doc in context_docs:
                            filename = os.path.basename(doc.metadata['source'])
                            ref_title = f":blue[Source document: {filename}]"
                            with st.popover(ref_title):
                                st.caption(doc.page_content)


if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()