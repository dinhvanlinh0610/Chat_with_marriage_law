from vector_store import ChromaDB
from embedding import HuggingFaceEmbedding
from utils import hyde_search, vector_search, init_llm
embedding_model = HuggingFaceEmbedding("keepitreal/vietnamese-sbert")
api_key = input("Enter your API key: ")
llm = init_llm(model_name = "llama3.2", api_key = api_key, type_llm = "offline")
# vector_store = ChromaDB(collection_name="test", embedding_model=embedding_model.embedding, persist_directory="./db/chroma_db")

# results = hyde_search(llm=llm,embedding_model=embedding_model.embedding,chroma=vector_store, query="covid", k=5)
# print(results)
