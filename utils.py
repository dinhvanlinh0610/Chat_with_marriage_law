from loader import PDFLoader
from chunker import SemanticChunk
from vector_store import ChromaDB
from llms import OnlineLLM
from llms.local_llms import run_ollama_container, run_ollama_model
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

def save_data(collection_name="test", embedding_model=None, path="./db/chroma_db", documents=None):

    
    vector_store = ChromaDB(collection_name=collection_name, embedding_model=embedding_model, persist_directory=path)
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
def init_llm(model_name = "llama3-8b-8192", api_key = None, type_llm = "online"):
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
    else:
        llm = init_local_llms()
    return llm

def RAG(llm, chroma, query, k= None, direct = None, type_search = "semantic"):

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
    if direct:
        if type_search == "vector":
            docs = chroma.query_directly(query, k, direct)
    


        return chroma.query_directly(query, k, direct)
    else:
        return llm.query(query, chroma, k)
