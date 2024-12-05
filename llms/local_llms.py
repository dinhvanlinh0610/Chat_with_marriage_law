import os
import platform
import requests
import subprocess

from dotenv import load_dotenv

load_dotenv()

ollama_endpoint = os.getenv("OLLAMA_ENDPOINT") or "http://localhost:11434"

OLLAMA_MODEL_ONLINE = {
    
    "Llama 3.2 (3B - 2.0GB)": "llama3.2",
    "Llama 3.2 (1B - 1.3GB)": "llama3.2:1b",
    "Llama 3.1 (8B - 4.7GB)": "llama3.1",
    "Llama 3.1 (70B - 40GB)": "llama3.1:70b",
    "Llama 3.1 (405B - 231GB)": "llama3.1:405b",
    "Phi 3 Mini (3.8B - 2.3GB)": "phi3",
    "Phi 3 Medium (14B - 7.9GB)": "phi3:medium",
    "Gemma 2 (2B - 1.6GB)": "gemma2:2b",
    "Gemma 2 (9B - 5.5GB)": "gemma2",
    "Gemma 2 (27B - 16GB)": "gemma2:27b",
    "Mistral (7B - 4.1GB)": "mistral",
    "Moondream 2 (1.4B - 829MB)": "moondream",
    "Neural Chat (7B - 4.1GB)": "neural-chat",
    "Starling (7B - 4.1GB)": "starling-lm",
    "Code Llama (7B - 3.8GB)": "codellama",
    "Llama 2 Uncensored (7B - 3.8GB)": "llama2-uncensored",
    "LLaVA (7B - 4.5GB)": "llava",
    "Solar (10.7B - 6.1GB)": "solar"
}

GGUF_MODEL_OPTIONS ={
    "Llama-3.2-1B-Instruct-GGUF": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
    "SmolLM-1.7B-Instruct-v0.2-GGUF": "hf.co/MaziyarPanahi/SmolLM-1.7B-Instruct-v0.2-GGUF",
}

# Function to install Nvidia Container Toolkit (for Nvidia GPU setup)
def install_nvidia_toolkit():
    # st.info("Installing NVIDIA Container Toolkit...")
    os.system("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg")
    os.system("curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list")
    os.system("sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit")
    os.system("sudo nvidia-ctk runtime configure --runtime=docker")
    os.system("sudo systemctl restart docker")

# Function to check if NVIDIA GPU is available
def has_nvidia_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Function to check if AMD GPU is available
def has_amd_gpu():
    try:
        result = subprocess.run(['lspci'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return 'AMD' in result.stdout.decode()
    except FileNotFoundError:
        return False    

def use_existing_or_run_container(container_name, image_name, gpu_option=None, position_noti="content"):
    """
    Function to reuse an existing container if available, otherwise start a new one.
    """
    # Check if the container exists
    result = subprocess.run(["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                            capture_output=True, text=True)
    container_exists = container_name in result.stdout.strip().split("\n")

    if container_exists:
        # Check if the container is running
        result = subprocess.run(["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                                capture_output=True, text=True)
        is_running = container_name in result.stdout.strip().split("\n")

        if is_running:
            if position_noti == "content":
                print(f"Container '{container_name}' is already running.")
            else:
                print(f"Container '{container_name}' is already running.")
            return  # Use the running container
        else:
            # Start the stopped container
            subprocess.run(["docker", "start", container_name])
            if position_noti == "content":
                print(f"Started existing container '{container_name}'.")
            else:
                print(f"Started existing container '{container_name}'.")
            return

    # If the container doesn't exist, create a new one
    if gpu_option == "nvidia":
        os.system(f"docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name {container_name} {image_name}")
    elif gpu_option == "amd":
        os.system(f"docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name {container_name} {image_name}")
    else:
        os.system(f"docker run -d -v ollama:/root/.ollama -p 11434:11434 --name {container_name} {image_name}")

    if position_noti == "content":
        print(f"Created and started a new container '{container_name}'.")
    else:
        print(f"Created and started a new container '{container_name}'.")

def remove_running_container(
        container_name,
        position_noti="content"
    ):
    # Check if the container is running
    result = subprocess.run(["docker", "ps", "-q", "--filter", f"name={container_name}"], capture_output=True, text=True)
    if result.stdout.strip():  # Container is running
        os.system(f"docker rm -f {container_name}")
        if position_noti == "content":
            print(f"Removed the running container '{container_name}'.")
            # st.success(f"Removed the running container '{container_name}'.")
        else:
            # st.sidebar.success(f"Removed the running container '{container_name}'.")
            print(f"Removed the running container '{container_name}'.")
# Function to run the Ollama container based on the hardware type
def run_ollama_container(position_noti="content"):
    """
    Function to start or reuse the Ollama container based on system and GPU configuration.
    """
    system = platform.system().lower()
    container_name = "ollama"
    image_name = "ollama/ollama"

    if system == "linux" or system == "darwin":  # macOS or Linux
        if has_nvidia_gpu():
            print("NVIDIA GPU detected. Ensuring NVIDIA Container Toolkit is installed...")
            install_nvidia_toolkit()  # Ensure NVIDIA toolkit is installed
            use_existing_or_run_container(container_name, image_name, gpu_option="nvidia", position_noti=position_noti)
        elif has_amd_gpu():
            print("AMD GPU detected. Starting with ROCm support...")
            use_existing_or_run_container(container_name, image_name, gpu_option="amd", position_noti=position_noti)
        else:
            print("No GPU detected. Starting with CPU-only support...")
            use_existing_or_run_container(container_name, image_name, position_noti=position_noti)
    elif system == "windows":
        print("Please ensure Docker Desktop is installed and running on Windows.")
        # subprocess.run(["docker", "run", "-d", "-v", "ollama:/root/.ollama", "-p", "11434:11434", "--name", container_name, image_name])
        use_existing_or_run_container(container_name, image_name, position_noti=position_noti)
import logging
import requests
from typing import List, Dict, Union

# Thiết lập ghi log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Định nghĩa các endpoint API dưới dạng hằng số
PULL_ENDPOINT = "/api/pull"
CHAT_ENDPOINT = "/api/chat"
GENERATE_ENDPOINT = "/api/generate"

class LocalLLM:
    def __init__(self, model_name: str, position_noti: str = "content"):
        self.model_name = model_name
        self.base_url = ollama_endpoint  # ollama_endpoint là địa chỉ API của mô hình
        self.position_noti = position_noti
        self.pull_model()

    def pull_model(self):
        logger.info(f"Đang tải mô hình '{self.model_name}'...")
        try:
            # Gửi yêu cầu POST để tải mô hình
            response = requests.post(
                f"{self.base_url}{PULL_ENDPOINT}",
                json={"model": self.model_name},
                timeout=10  # Thêm timeout để tránh request treo quá lâu
            )

            # Kiểm tra mã trạng thái HTTP
            if response.status_code == 200:
                if self.position_noti == "content":
                    print(f"Tải mô hình '{self.model_name}' thành công!")
                else:
                    print(f"Tải mô hình '{self.model_name}' thành công!")
            else:
                error_message = response.json().get("error", "Có lỗi xảy ra.")
                raise ValueError(f"Tải mô hình '{self.model_name}' thất bại: {error_message}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi khi gửi yêu cầu: {e}")
            raise

    def generate_content(self, messages: List[Dict[str, str]]) -> Union[Dict[str, Union[str, int]], None]:
        try:
            # Dữ liệu gửi đến API
            data = {
                "model": self.model_name,
                "messages": [
                { "role": "system",
                        "content": """Bạn là một chuyên gia pháp lý chuyên về Luật Hôn nhân và Gia đình Việt Nam. Tôi sẽ cung cấp câu hỏi và các đoạn văn bản tài liệu.
Nhiệm vụ của bạn:
1. Sử dụng nội dung trong tài liệu đã cung cấp để trả lời câu hỏi.
2. Trích dẫn chính xác số điều, khoản, mục và nội dung luật từ tài liệu(tự bổ sung hoặc sửa chữa).
3. Nếu câu trả lời nằm ở nhiều điều khoản khác nhau, liệt kê tất cả, nhưng không mở rộng thêm bất kỳ ý kiến hay giải thích nào.
4. Nếu không tìm thấy câu trả lời trong tài liệu, chỉ trả lời: 'Không tìm thấy quy định phù hợp trong tài liệu được cung cấp.'"""},
    {"role": "user", "content": messages}
  ],
                "stream": False
            }

            # Gửi yêu cầu POST để trò chuyện với mô hình
            response = requests.post(
                f"{self.base_url}{CHAT_ENDPOINT}",
                json=data,
                timeout=3000
            )

            # Kiểm tra mã trạng thái HTTP
            if response.status_code == 200:
                response_json = response.json()

                assistant_message = response_json.get("message", {}).get("content", "")

                return {
                    "content": assistant_message,
                    "model": response_json.get('model'),
                    "created_at": response_json.get('created_at'),
                    "total_duration": response_json.get('total_duration'),
                    "load_duration": response_json.get('load_duration'),
                    "prompt_eval_count": response_json.get('prompt_eval_count'),
                    "prompt_eval_duration": response_json.get('prompt_eval_duration'),
                    "eval_count": response_json.get('eval_count'),
                    "eval_duration": response_json.get('eval_duration'),
                    "done": response_json.get('done')
                }
            else:
                logger.error(f"Tạo cuộc trò chuyện với mô hình '{self.model_name}' thất bại")
                return None

        except Exception as e:
            logger.error(f"Lỗi khi trò chuyện: {str(e)}")
            return None

    def generate_content2(self, prompt: str) -> str:
        try:
            # Dữ liệu gửi đến API
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
            }

            # Gửi yêu cầu POST để sinh nội dung
            response = requests.post(
                f"{self.base_url}{GENERATE_ENDPOINT}",
                json=data,
                timeout=10
            )

            # Kiểm tra mã trạng thái HTTP
            if response.status_code == 200:
                response_json = response.json()
                return response_json.get("response", "")
            else:
                logger.error(f"Thất bại khi tạo nội dung với mô hình '{self.model_name}'")
                return ""

        except Exception as e:
            logger.error(f"Lỗi khi tạo nội dung: {str(e)}")
            return ""

def run_ollama_model(
        model_name="gemma2:2b",
        position_noti="content"
    ):
    # Check if the Ollama server is running
    try:
        response = requests.get(ollama_endpoint)
        if response.status_code != 200:
            if position_noti == "content":
                print("Ollama server is not running. Please start the server first.")
                # st.error("Ollama server is not running. Please start the server first.")
            else:
                print("Ollama server is not running. Please start the server first.")
                # st.sidebar.error("Ollama server is not running. Please start the server first.")
            return None
    except requests.ConnectionError:
        if position_noti == "content":
            print("Ollama server is not reachable. Please check if it's running.")
            # st.error("Ollama server is not reachable. Please check if it's running.")
        else:
            print("Ollama server is not reachable. Please check if it's running.")
            # st.sidebar.error("Ollama server is not reachable. Please check if it's running.")
        return None

    # Create and return an instance of LocalLlms
    return LocalLLM(
        model_name,
        position_noti=position_noti
    )
