from unittest.mock import Mock, patch, MagicMock
from lib.utils import display_msg, print_qa, format_docs

class TestUtils:
    @patch('lib.utils.st')
    def test_display_msg_user(self, mock_st):
        """Test displaying a user message"""
        # Arrange
        mock_st.session_state.messages = []
        mock_chat_message = MagicMock()
        mock_st.chat_message.return_value = mock_chat_message

        msg = "Hello, how are you?"
        author = "user"

        # Act
        display_msg(msg, author)

        # Assert
        assert len(mock_st.session_state.messages) == 1
        assert mock_st.session_state.messages[0] == {"role": "user", "content": msg}
        mock_st.chat_message.assert_called_once_with("user")
        mock_chat_message.write.assert_called_once_with(msg)

    @patch('lib.utils.st')
    def test_display_msg_assistant(self, mock_st):
        """Test displaying an assistant message"""
        # Arrange
        mock_st.session_state.messages = []
        mock_chat_message = MagicMock()
        mock_st.chat_message.return_value = mock_chat_message

        msg = "I'm doing well, thank you!"
        author = "assistant"

        # Act
        display_msg(msg, author)

        # Assert
        assert len(mock_st.session_state.messages) == 1
        assert mock_st.session_state.messages[0] == {"role": "assistant", "content": msg}
        mock_st.chat_message.assert_called_once_with("assistant")

    @patch('lib.utils.logger')
    def test_print_qa(self, mock_logger):
        """Test print_qa logs the correct information"""

        # Arrange
        class MockClass:
            pass

        question = "What is the capital of France?"
        answer = "Paris"

        # Act
        print_qa(MockClass, question, answer)

        # Assert
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "MockClass" in call_args
        assert question in call_args
        assert answer in call_args

    def test_format_docs_empty_list(self):
        """Test format_docs with empty list"""
        # Arrange
        docs = []

        # Act
        result = format_docs(docs)

        # Assert
        assert result == ""

    def test_format_docs_single_doc(self):
        """Test format_docs with single document"""
        # Arrange
        mock_doc = Mock()
        mock_doc.page_content = "This is a test document."
        docs = [mock_doc]

        # Act
        result = format_docs(docs)

        # Assert
        assert result == "This is a test document."

    def test_format_docs_multiple_docs(self):
        """Test format_docs with multiple documents"""
        # Arrange
        mock_doc1 = Mock()
        mock_doc1.page_content = "First document content."
        mock_doc2 = Mock()
        mock_doc2.page_content = "Second document content."
        mock_doc3 = Mock()
        mock_doc3.page_content = "Third document content."
        docs = [mock_doc1, mock_doc2, mock_doc3]

        # Act
        result = format_docs(docs)

        # Assert
        expected = "First document content.\n\nSecond document content.\n\nThird document content."
        assert result == expected