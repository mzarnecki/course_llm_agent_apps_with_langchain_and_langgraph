from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self):
        self.documents_path = ''

    def load_chunks(self)->list[Document]:
        """Load and chunk text documents from the data directory.

        Returns:
            list[Document]: A list of Document objects
        """
        docs = []
        files = []
        data_path = Path("data")

        for file in data_path.iterdir():
            if file.suffix == ".txt":
                with open(file) as f:
                    docs.append(f.read())
                    files.append(file.name)

        # Split documents and store in vector db
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = []
        for i, doc in enumerate(docs):
            for chunk in text_splitter.split_text(doc):
                splits.append(Document(page_content=chunk, metadata={"source": files[i]}))

        return splits