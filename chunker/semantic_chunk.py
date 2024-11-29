from langchain_experimental.text_splitter import SemanticChunker

class SemanticChunk():
    def __init__(self, embedding_model):
        """
        Initialize the SemanticChunk object.

        Args:
            embedding_model (str): embedding model used to split text into semantic chunks.
        """
        self.embedding_model = embedding_model
        self.text_splitter = SemanticChunker(embeddings=self.embedding_model, breakpoint_threshold_type="percentile")
        self.docs = []

    def splits(self, documents):
        """
        Split documents into semantic chunks.

        Args:
            docs (list): List of documents to be split.

        Returns:
            list: List of semantic chunks.
        """
        self.docs = self.text_splitter.split_documents(documents=documents)
        return self.docs
