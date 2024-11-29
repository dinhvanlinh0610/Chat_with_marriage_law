from langchain_chroma import Chroma
from uuid import uuid4

class ChromaDB():

    def __init__(self, collection_name="example_collection", embedding_model=None, persist_directory="./db/chroma_db"):
        """
        Initialize the ChromaDB object.

        Args:
            collection_name (str): The name of the collection to be created.
            embedding_model (str): The embedding model to be used.
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.chroma = Chroma(collection_name=self.collection_name, embedding_function=self.embedding_model, persist_directory=self.persist_directory)
        
        
    def create_retriever(self):
        """
        Create a retriever object.

        Returns:
            object: A retriever object.
        """
        retriever = self.chroma.as_retriever(
            search_type="hybrid",
            search_kwargs={"k": 3, "fetch_k": 5}
        )
        return retriever
    
    def add(self, documents):
        """
        Add documents to the collection.

        Args:
            documents (list): List of documents to be added.

        Returns:
            object: A Chroma object.

        """
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.chroma.add_documents(documents=documents, uuids=uuids)
        return self.chroma
    
    def query_directly(self, query, k , direct):
        """
        Query documents from the collection.

        Args:
            query (str): The query to be searched.
            k (int): The number of documents to be returned.
            direct (str): The source of the document.

        Returns:
            list: List of documents.

        """
        results = self.chroma.similarity_search(
            query=query,
            k=k,
            filter={"source": direct}
        )

        return results
    def query_with_score(self, query, k):
        """
        Query documents from the collection.

        Args:
            query (str): The query to be searched.
            k (int): The number of documents to be returned.

        Returns:
            list: List of documents.
        """
        results = self.chroma.similarity_search(
            query=query,
            k=k
        )

        return results
