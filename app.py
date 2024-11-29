from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    """
    This is a function to return a message. Introduction about chatbot and how to use it.

    Returns:
    str: A message to introduce chatbot.

    """
    return {"message": "Welcome to the chatbot! Please enter a message to start a conversation."}

