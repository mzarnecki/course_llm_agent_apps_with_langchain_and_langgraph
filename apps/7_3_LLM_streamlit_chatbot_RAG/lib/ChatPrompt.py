from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

class ChatPrompt:
    def get_prompt(self)->ChatPromptTemplate:
        """Crete prompt template with system prompt, placeholder for chat history, and user question.

        Returns:
            ChatPromptTemplate: LangChain template for chatbot prompt.
        """
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a chatbot tasked with responding to questions based on attached documents content.\n"
             "Use the following pieces of retrieved context to answer the question. "
             "If the question references previous conversation, use the chat history to understand the context.\n"
             "Depend only on source documents.\n\n"
             "Context:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])