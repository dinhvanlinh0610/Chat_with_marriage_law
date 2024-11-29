# Import API
from fastapi import FastAPI

# Import Utils
from utils import process_data

# Import NLP

### Initialize FastAPI
app = FastAPI()


### Endpoint ###


@app.get("/")
def read_root():
    """
    This is a function to return a message. Introduction about chatbot and how to use it.

    Returns:
    str: A message to introduce chatbot.

    """
    return {"message": "Welcome to the chatbot! Please enter a message to start a conversation."}

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
    return process_data(data_path)

