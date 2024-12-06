import os
import requests
from typing import List, Dict, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default Ollama endpoint
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT") or "http://localhost:11434"

# API endpoints for Ollama
PULL_ENDPOINT = "/api/pull"
CHAT_ENDPOINT = "/api/chat"
GENERATE_ENDPOINT = "/api/generate"

class LocalLLM:
    def __init__(self, model_name: str, position_noti: str = "content"):
        """
        Initialize the LocalLLM instance.
        Args:
            model_name (str): The name of the model to use.
            position_noti (str): Position notification type. Defaults to "content".
        """
        self.model_name = model_name
        self.base_url = OLLAMA_ENDPOINT  # Ollama local endpoint
        self.position_noti = position_noti
        self.pull_model()

    def pull_model(self):
        """
        Pull the specified model from Ollama local.
        """
        print(f"Pulling model '{self.model_name}' from Ollama local...")
        try:
            response = requests.post(
                f"{self.base_url}{PULL_ENDPOINT}",
                json={"model": self.model_name},
                timeout=10
            )

            if response.status_code == 200:
                print(f"Successfully pulled model '{self.model_name}'!")
            else:
                error_message = response.json().get("error", "An unknown error occurred.")
                raise ValueError(f"Failed to pull model '{self.model_name}': {error_message}")

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            raise

    def generate_content(self, messages):
        """
        Generate content based on provided messages using the model.
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries to send to the model.
        Returns:
            dict or None: The response content and metadata if successful, None otherwise.
        """
        try:
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": """Bạn là một chuyên gia pháp lý chuyên về Luật Hôn nhân và Gia đình Việt Nam. Tôi sẽ cung cấp câu hỏi và các đoạn văn bản tài liệu. Nhiệm vụ của bạn: 
                    1. Sử dụng nội dung trong tài liệu đã cung cấp để trả lời câu hỏi. 
                    2. Trích dẫn chính xác số điều, khoản, mục và nội dung luật từ tài liệu (tự bổ sung hoặc sửa chữa). 
                    3. Nếu câu trả lời nằm ở nhiều điều khoản khác nhau, liệt kê tất cả, nhưng không mở rộng thêm bất kỳ ý kiến hay giải thích nào. 
                    4. Nếu không tìm thấy câu trả lời trong tài liệu, chỉ trả lời: 'Không tìm thấy quy định phù hợp trong tài liệu được cung cấp.'"""},
                    {"role": "user", "content": messages}
                ] ,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}{CHAT_ENDPOINT}",
                json=data,
                timeout=300  # Increased timeout to handle large models
            )

            if response.status_code == 200:
                response_json = response.json()
                assistant_message = response_json.get("message", {}).get("content", "")
                return {
                    "content": assistant_message,
                    "model": response_json.get("model"),
                    "created_at": response_json.get("created_at"),
                    "total_duration": response_json.get("total_duration"),
                    "load_duration": response_json.get("load_duration"),
                    "prompt_eval_count": response_json.get("prompt_eval_count"),
                    "prompt_eval_duration": response_json.get("prompt_eval_duration"),
                    "eval_count": response_json.get("eval_count"),
                    "eval_duration": response_json.get("eval_duration"),
                    "done": response_json.get("done")
                }
            else:
                error_message = response.json().get("error", "An unknown error occurred.")
                print(f"Chat generation failed: {error_message}")
                return None

        except Exception as e:
            print(f"Error during chat generation: {e}")
            return None

    def generate_content_answer(self, prompt: str) -> str:
        """
        Generate content directly from a prompt using the model.
        Args:
            prompt (str): The prompt string to generate content from.
        Returns:
            str: The generated text, or an empty string if an error occurred.
        """
        try:
            data = {
                "model": self.model_name,
                "prompt": f"Nhận vào 1 câu hỏi và các tài liệu liên quan, lọc để trả về các tài liệu hữu dụng để trả lời câu hỏi đó : {prompt}",
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}{GENERATE_ENDPOINT}",
                json=data,
                timeout=300
            )

            if response.status_code == 200:
                return response.json().get("generated_text", "")
            else:
                error_message = response.json().get("error", "An unknown error occurred.")
                print(f"Content generation failed: {error_message}")
                return ""
        except Exception as e:
            print(f"Error during content generation: {e}")
            return ""
