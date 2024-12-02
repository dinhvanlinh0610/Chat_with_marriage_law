# Import API
from fastapi import FastAPI

# Import Utils
from utils import process_data, split_data, save_data, init_llm

# Import Embedding Model
from embedding import HuggingFaceEmbedding

### Initialize FastAPI
app = FastAPI()

# Initialize chat session
session_state = {
    "embedding_model": None,
    "llm": None,
    "chroma": None,
    "collection_name": None,
    "history_chat": []

}

### Endpoint ###


@app.get("/")
def read_root():
    """
    This is a function to return a message. Introduction about chatbot and how to use it.

    Returns:
    str: A message to introduce chatbot.

    """

    return {"message": "Welcome to the chatbot! Please enter a message to start a conversation."}

@app.get("/initialize/")
def initialize(model_name: str, api_key: str, type_llm: str):
    """
    This is a function to initialize the model.

    Returns:
    str: A message to show that the model is initialized.

    """
    try:
        session_state["llm"] = init_llm(model_name, api_key, type_llm)
    except Exception as e:
        # Log error
        return {"error": str(e)}
    return "Model is initialized!"

# endpoint nhận vào data_path và trả về danh sách các câu trong file pdf
@app.get("/upload_file/")
def upload_file(data_path: str):
    """
    This is a function to process the data from the client.

    Args:
        data_path (str): Đường dẫn đến file pdf.

    Returns:
        list: Danh sách các câu trong file pdf.

    """
    embedding_model = HuggingFaceEmbedding("keepitreal/vietnamese-sbert")
    session_state["embedding_model"] = embedding_model

    documents = process_data(data_path)
    docs = split_data(documents, embedding_model.embedding)
    chroma = save_data(collection_name="test", embedding_model=embedding_model.embedding, documents=docs)
    session_state["chroma"] = chroma
    return "success"

@app.get("/setup/")
def setup(type_search: str):
    """
    This is a function to setup the chatbot.

    Returns:
    str: A message to show that the chatbot is setup.

    """
    if type_search == "semantic":
        return "Setup semantic search!"

@app.post("/chat/")
def chatbot(message: str):

    llm = session_state["llm"]
    response = llm.generate_content(message)
    return {"response": response}


