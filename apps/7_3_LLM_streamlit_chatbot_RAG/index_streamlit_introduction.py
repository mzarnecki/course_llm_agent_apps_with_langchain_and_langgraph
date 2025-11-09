import streamlit as st

st.set_page_config(page_title="Simple Chat", page_icon="💬")
st.header('Simple Streamlit Chat')
st.write('A basic chat interface demonstrating Streamlit chat components.')


def initialize_chat_history():
    """Initialize chat history in session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

def display_chat_history():
    """Display all messages from chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def handle_user_input(user_query: str):
    """Handle user input and generate response"""
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Simulate a simple response (replace this with your own logic)
    response = f"You said: '{user_query}'. This is a simple echo response!"

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    # Initialize chat history
    initialize_chat_history()

    # Display existing chat history
    display_chat_history()

    # Chat input at the bottom
    user_query = st.chat_input(placeholder="Type your message here...")

    if user_query:
        handle_user_input(user_query)


if __name__ == "__main__":
    main()