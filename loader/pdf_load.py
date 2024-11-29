from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PDFMinerLoader

class PDFLoader():
    def __init__(self, path):
        """
        Initialize the PDFLoader object.
        
        Args:
            path (str): The path to the PDF file to be loaded.

        """
        self.path = path
        self.pdf_loader = PDFMinerLoader(self.path)
        self.docs = []

    def loads(self):
        self.docs = self.pdf_loader.load()
        return self.docs