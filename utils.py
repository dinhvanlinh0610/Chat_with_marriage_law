from loader import PDFLoader
from chunker import SemanticChunk
from vector_store import ChromaDB
from llms import OnlineLLM
import numpy as np
from llms.local_llms import run_ollama_container, run_ollama_model
from llms.local_llms_linux import LocalLLM
# Hàm xử lý dữ liệu nhận được từ client 
def process_data(data):
    """

    This function takes in a data_path and checks if the data_path is a pdf file.
    If it is a pdf file, use PDFLoader to load data from the pdf file.
    If not, return an error message.

    Args:
        data (str): Đường dẫn đến file pdf.

    Returns:
        list: Danh sách các câu trong file pdf.

    """
    if data.endswith('.pdf'):
        pdf_loader = PDFLoader(data)
        docs = pdf_loader.loads()
        return docs
    else:
        return "Invalid file format. Please upload a PDF file."

def split_data(data, embedding_model):
    """

    This function takes in a list of documents and splits them into semantic chunks.

    Args:
        data (list): List of documents.

    Returns:
        list: List of semantic chunks.

    """
    semantic_chunk = SemanticChunk(embedding_model=embedding_model)
    return semantic_chunk.splits(data)

def init_vector_store(collection_name="new", embedding_model=None, path="./db/chroma_db"):

    chroma = ChromaDB(collection_name=collection_name, embedding_model=embedding_model, persist_directory=path)
    return chroma

def save_data(vector_store, documents=None):

    
    chroma = vector_store.add(documents=documents)
    return chroma

def init_local_llms():
    print("Run ollama container")
    run_ollama_container()

    print("select model")

    # cho chọn model bằng list(OLLAMA_MODEL_OPTIONS.keys())
    selected_model = "llama3.2"

    print("Run model")
    localLLMs = run_ollama_model(selected_model)

    return localLLMs
def init_llm(model_name = None, api_key = None, type_llm = "online"):
    """
    This function initializes the language model.

    Args:
        model_name (str): The name of the model.
        api_key (str): The API key.
        type_llm (str): The type of language model.

    Returns:
        object: The language model.
    """
    if type_llm == "online":
        llm = OnlineLLM(model_name, api_key)
    elif type_llm == "local_windows":
        llm = init_local_llms()
    else:
        llm = LocalLLM(model_name)
    return llm

def RAG(llm, chroma, query, k= None, direct = None, type_search = "retriever_search", embedding_model=None ):

    """
    This function takes in a query and returns a response using the RAG model.

    Args:
        llm (object): The language model.
        chroma (object): The vector store.
        query (str): The query to be searched.
        k (int): The number of documents to be returned.
        direct (str): The source of the document.

    Returns:
        list: List of documents.

    """
   
    if type_search == "vector_search":
        docs = chroma.query_by_vector(query, k)
    elif type_search == "retriever_search":
        docs = retriever_search(chroma, query)
    elif type_search == "keyword_search":
        docs = keyword_search(chroma, query, k)
    elif type_search == "hyde_search":
        docs = hyde_search(llm, embedding_model, query, chroma, k, num_samples=1)
    else:
        docs = llm.generate_content(query)

    return docs
 

def retriever_search(chroma, query):

    retriever = chroma.create_retriever()
    results = retriever.invoke(query)
    retrieval_docs = ""

    i=0
    for result in results:
        i+=1
        retrieval_docs += f"\n{i})"
        if result.page_content:
            retrieval_docs += f"{result.page_content}\n"

        retrieval_docs += "\n"

    return retrieval_docs
    

def vector_search(chroma, query, k):

    results = chroma.query_by_vector(query, k =k)

    retrieval_docs = ""

    i=0
    for result in results:
        i+=1
        retrieval_docs += f"\n{i})"
        if result.page_content:
            retrieval_docs += f"{result.page_content}\n"

        retrieval_docs += "\n"

    return retrieval_docs

def keyword_search(chroma, query, k):

    results = chroma.query_with_score(query, k =k)

    retrieval_docs = ""

    i=0
    for result in results:
        i+=1
        retrieval_docs += f"\n{i})"
        if result.page_content:
            retrieval_docs += f"{result.page_content}\n"

        retrieval_docs += "\n"

    return retrieval_docs

def generate_hypothetical_documents(llm,query,num_samples=1):
    """
    This function generates hypothetical documents using the language model.

    Args:
        llm (object): The language model.
        query (str): The query to be searched.
        num_samples (int): The number of samples to be generated.

    Returns:
        list: List of hypothetical documents.

    """
    hypothetical_docs = []

    for _ in range(num_samples):
        enhanced_prompt = f"Viết đoạn văn trả lời câu hỏi: {query}. Câu trả lời nên liên quan đến vấn đề và có trong Luật Hôn nhân Việt Nam 2014."
        response = llm.generate_content(enhanced_prompt)
        hypothetical_docs.append(response)

    return hypothetical_docs

def encode_hypothetical_documents(embedding_model, hypothetical_docs):
    
    print(hypothetical_docs)
    embeddings = [embedding_model.embed_query(doc) for doc in hypothetical_docs]

    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def hyde_search(llm, embedding_model, query, chroma, k, num_samples=1):

    hypothetical_docs = generate_hypothetical_documents(llm, query, num_samples)
    avg_embedding = encode_hypothetical_documents(embedding_model, hypothetical_docs)

    results = chroma.similarity_search_by_vector(embedding = avg_embedding, k=k)

    retrieval_docs = ""

    i=0
    for result in results:
        i+=1
        retrieval_docs += f"\n{i})"
        if result.page_content:
            retrieval_docs += f"{result.page_content}\n"

        retrieval_docs += "\n"

    return retrieval_docs