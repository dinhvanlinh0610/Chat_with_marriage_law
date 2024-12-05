# Import API
from fastapi import FastAPI

# Import Utils
from utils import process_data, split_data, save_data, init_llm, RAG, init_vector_store

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
        session_state["embedding_model"] = HuggingFaceEmbedding("keepitreal/vietnamese-sbert")
        session_state["chroma"] = init_vector_store(collection_name="new", embedding_model=session_state["embedding_model"].embedding, path="./db/chroma_db")
    except Exception as e:
        # Log error
        return {"error": str(e)}
    return "Model is initialized!"

# endpoint nhận vào data_path và trả về danh sách các câu trong file pdf
@app.get("/upload_file/")
def upload_file(data_path: str):
    """
    This is a function to upload the file.

    Args:
        data_path (str): The path to the file.

    Returns:
        str: A message to show that the file is uploaded.


    """
    embedding_model = HuggingFaceEmbedding("keepitreal/vietnamese-sbert")
    session_state["embedding_model"] = embedding_model

    documents = process_data(data_path)
    docs = split_data(documents, embedding_model.embedding)
    vector_store = session_state["chroma"]
    chroma = save_data(vector_store, documents=docs)
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
    embedding_model =  session_state["embedding_model"]
    chroma = session_state["chroma"]
    # embedding_model = session_state["embedding_model"]
    retriever_docs = RAG(llm=llm,embedding_model=embedding_model.embedding, chroma=chroma, query=message, k=3)
    print("\n retriever_docs: ", retriever_docs, "\n")
    enhanced_prompt = """Câu hỏi: "{} \n Tài liệu: {}""".format(message, retriever_docs)

    response = llm.generate_content(enhanced_prompt)
    print("\n response: ", response, "\n")
    # enhanced = """Câu hỏi: "{} \n Câu trả lời: {}""".format(message, response)

    # perfect_answer = llm.generate_perfect_answer(enhanced)
    # print("\n perfect_answer: ", perfect_answer, "\n")
    return {"response": response}


