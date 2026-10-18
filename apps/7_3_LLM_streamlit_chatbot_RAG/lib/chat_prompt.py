from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class ChatPrompt:
    def get_prompt(self) -> ChatPromptTemplate:
        """Create prompt template with a system prompt, chat history, and question.

        Returns:
            ChatPromptTemplate: LangChain template for chatbot prompt.
        """
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a chatbot tasked with responding to questions based on "
             "attached documents content.\n"
             "Use the following pieces of retrieved context to answer the question. "
             "If the question references previous conversation, use the chat "
             "history to understand the context.\n"
             "Depend only on source documents.\n\n"
             "Context:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
