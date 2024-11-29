from langchain_huggingface.embeddings import HuggingFaceEmbeddings

class HuggingFaceEmbedding():
    def __init__(self, model_name):
        """
        Initialize the HuggingFaceEmbedding object.

        Args:
            model_name (str): The name of the Hugging Face model to be used.
        """
        self.model_name = model_name
        self.embedding = HuggingFaceEmbeddings(model_name=self.model_name)
        self.docs = []

    def embed(self, documents):
        """
        Embed documents using the Hugging Face model.

        Args:
            docs (list): List of documents to be embedded.

        Returns:
            list: List of embedded documents.
        """
        self.docs = self.embedding.embed_documents(documents=documents)
        return self.docs