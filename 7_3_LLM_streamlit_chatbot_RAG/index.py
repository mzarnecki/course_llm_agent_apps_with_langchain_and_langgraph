import os
from lib import utils
import streamlit as st

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        # Store for session histories
        if 'langchain_store' not in st.session_state:
            st.session_state.langchain_store = {}
        self.store = st.session_state.langchain_store

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
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

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    @st.spinner('Analyzing documents..')
    def import_source_documents(self):
        # load documents
        docs = []
        files = []
        for file in os.listdir("data"):
            if file.endswith(".txt"):
                with open(os.path.join("data", file)) as f:
                    docs.append(os.path.join("data", f.read()))
                    files.append(file)

        # Split documents and store in vector db
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = []
        for i, doc in enumerate(docs):
            for chunk in text_splitter.split_text(doc):
                splits.append(Document(page_content=chunk, metadata={"source": files[i]}))

        vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 2, 'fetch_k': 4}
        )

        # Contextualize question prompt - reformulates question based on history
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question "
             "which might reference context in the chat history, "
             "formulate a standalone question which can be understood "
             "without the chat history. Do NOT answer the question, "
             "just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a chatbot tasked with responding to questions based on attached documents content.\n"
             "Use the following pieces of retrieved context to answer the question. "
             "Depend only on source documents.\n\n"
             "Context:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Function to handle retrieval with or without history
        def contextualized_retrieval(input_dict):
            if input_dict.get("chat_history"):
                # Use history-aware retriever
                standalone_question = (
                        contextualize_q_prompt
                        | self.llm
                        | StrOutputParser()
                ).invoke({
                    "input": input_dict["input"],
                    "chat_history": input_dict["chat_history"]
                })
                docs = retriever.invoke(standalone_question)
            else:
                # Direct retrieval
                docs = retriever.invoke(input_dict["input"])
            return docs

        # Create the retrieval chain that returns both docs and formatted context
        retrieval_chain = RunnablePassthrough.assign(
            context=contextualized_retrieval
        ) | RunnablePassthrough.assign(
            formatted_context=lambda x: self.format_docs(x["context"])
        )

        # Complete RAG chain using LCEL
        rag_chain = (
                retrieval_chain
                | RunnablePassthrough.assign(
            answer=(
                    {
                        "context": itemgetter("formatted_context"),
                        "input": itemgetter("input"),
                        "chat_history": itemgetter("chat_history")
                    }
                    | qa_prompt
                    | self.llm
                    | StrOutputParser()
            )
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

            qa_chain = self.import_source_documents()

            utils.display_msg(user_query, 'user')

            # Add user message to LangChain history
            session_history.add_message(HumanMessage(content=user_query))

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_text = ""
                context_docs = None

                # Stream the response chunks
                for chunk in qa_chain.stream(
                        {"input": user_query},
                        config={
                            "configurable": {"session_id": "default_session"}
                        }
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