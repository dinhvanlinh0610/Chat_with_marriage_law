from loader import PDFLoader
from chunker import SemanticChunk
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