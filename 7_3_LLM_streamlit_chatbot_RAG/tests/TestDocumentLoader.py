import pytest
from lib.DocumentLoader import DocumentLoader
from langchain_core.documents import Document

class TestDocumentLoader:

    @pytest.fixture
    def loader(self):
        """Create a DocumentLoader instance for testing."""
        return DocumentLoader()

    @pytest.fixture
    def mock_data_path(self, tmp_path):
        """Create a temporary data directory with test files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create test .txt files
        (data_dir / "test1.txt").write_text("This is test document one with some content.")
        (data_dir / "test2.txt").write_text("This is test document two with different content.")
        (data_dir / "ignore.pdf").write_text("Should be ignored")

        return data_dir

    def test_init(self, loader):
        """Test DocumentLoader initialization."""
        assert loader.documents_path == ''

    def test_load_chunks_basic(self, loader, tmp_path, monkeypatch):
        """Test basic document loading and splitting."""
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("Short document content.")

        # Change to tmp_path as working directory
        monkeypatch.chdir(tmp_path)

        # Execute
        splits = loader.load_chunks()

        # Assert
        assert len(splits) > 0
        assert all(isinstance(doc, Document) for doc in splits)
        assert all(hasattr(doc, 'page_content') for doc in splits)
        assert all(hasattr(doc, 'metadata') for doc in splits)

    def test_load_chunks_multiple_files(self, loader, tmp_path, monkeypatch):
        """Test loading multiple text files."""
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc1.txt").write_text("First document.")
        (data_dir / "doc2.txt").write_text("Second document.")
        (data_dir / "doc3.txt").write_text("Third document.")

        monkeypatch.chdir(tmp_path)

        # Execute
        splits = loader.load_chunks()

        # Assert
        sources = {doc.metadata['source'] for doc in splits}
        assert 'doc1.txt' in sources
        assert 'doc2.txt' in sources
        assert 'doc3.txt' in sources

    def test_load_chunks_filters_non_txt_files(self, loader, tmp_path, monkeypatch):
        """Test that only .txt files are loaded."""
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "valid.txt").write_text("Valid content.")
        (data_dir / "invalid.pdf").write_text("Should be ignored.")
        (data_dir / "invalid.docx").write_text("Should be ignored.")

        monkeypatch.chdir(tmp_path)

        # Execute
        splits = loader.load_chunks()

        # Assert
        sources = {doc.metadata['source'] for doc in splits}
        assert 'valid.txt' in sources
        assert 'invalid.pdf' not in sources
        assert 'invalid.docx' not in sources

    def test_load_chunks_empty_directory(self, loader, tmp_path, monkeypatch):
        """Test loading from empty data directory."""
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        monkeypatch.chdir(tmp_path)

        # Execute
        splits = loader.load_chunks()

        # Assert
        assert splits == []

    def test_load_chunks_chunking(self, loader, tmp_path, monkeypatch):
        """Test that large documents are properly chunked."""
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create a large document that will be split
        large_content = "A" * 2500  # Larger than chunk_size of 1000
        (data_dir / "large.txt").write_text(large_content)

        monkeypatch.chdir(tmp_path)

        # Execute
        splits = loader.load_chunks()

        # Assert - should have multiple chunks from single file
        assert len(splits) > 1
        assert all(doc.metadata['source'] == 'large.txt' for doc in splits)

    def test_load_chunks_metadata_preserved(self, loader, tmp_path, monkeypatch):
        """Test that source metadata is correctly preserved in splits."""
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "source.txt").write_text("Content here.")

        monkeypatch.chdir(tmp_path)

        # Execute
        splits = loader.load_chunks()

        # Assert
        for doc in splits:
            assert 'source' in doc.metadata
            assert doc.metadata['source'] == 'source.txt'

    def test_load_chunks_no_data_directory(self, loader, tmp_path, monkeypatch):
        """Test behavior when data directory doesn't exist."""
        monkeypatch.chdir(tmp_path)

        # Execute & Assert
        with pytest.raises(FileNotFoundError):
            loader.load_chunks()