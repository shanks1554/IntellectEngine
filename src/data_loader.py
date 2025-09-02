import pickle
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from logger import get_logger

class DataLoader:
    def __init__(self, raw_dir = 'data/raw', processed_dir = 'data/processed'):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_file = self.processed_dir/"processed_documents.pkl"
        self.logger = get_logger(self.__class__.__name__)
    
    def load_pdfs(self):
        """
        Load PDFs from raw_dir.
        If processed file exists, load from it instead of reprocessing.
        """
        if self.processed_file.exists():
            self.logger.info(f"Processed file already exists at {self.processed_file}. Skipping loading.")
            with open(self.processed_file, "rb") as f:
                return pickle.load(f)
            
        self.logger.info(f"Loading PDFs from {self.raw_dir}...")
        documents = []

        for pdf_file in self.raw_dir.glob("*.pdf"):
            try:
                self.logger.info(f"Loading: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                self.logger.error(f"Failed to load {pdf_file.name}: {e}")
        
        # Ensure processed directory exists
        self.processed_dir.mkdir(parents = True, exist_ok = True)

        # Save documents
        with open(self.processed_file, 'wb') as f:
            pickle.dump(documents, f)
            self.logger.info(f"Saved {len(documents)} documents to {self.processed_file}")
        
        return documents