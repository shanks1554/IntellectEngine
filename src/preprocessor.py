import pickle
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
from data_loader import DataLoader
from logger import get_logger


loader = DataLoader(raw_dir='data/raw', processed_dir='data/processed')
class Preprocessor:
    def __init__(self, processed_dir = 'data/processed', chunk_size = 1000, overlap = 200, loader = loader):
        self.processed_dir = Path(processed_dir)
        self.chunks_file = self.processed_dir/"chunks.pkl"
        self.metadata_file = self.processed_dir/"metadata.pkl"
        self.chunks_size = chunk_size
        self.overlap = overlap
        self.loader = loader
        self.logger = get_logger(self.__class__.__name__)

        # Ensure NLTK punkt is available
        try:
            nltk.data.find("tokenizers/punkt")
        except:
            nltk.download("punkt")
    def loader_method(self):
        if self.chunks_file.exists() and self.metadata_file.exists():
            self.logger.info("Chunks and metadata already exists. Skipping preprocessing...")
            with open(self.chunks_file,"rb") as f1, open(self.metadata_file, 'rb') as f2:
                chunks = pickle.load(f1)
                metadata = pickle.load(f2)
                return chunks, metadata
        self.logger.info("Processed chunks not found. Loading raw documents vai DataLoader...")
        documents = self.loader.load_pdfs()
        return self.process_documents(documents)
        
    def process_documents(self, documents):
        """
        Preprocess documents into chunks.
        """
        self.logger.info("Started preprocessing of documents...")
        chunks, metadata = [], []

        for idx, doc in enumerate(documents):
            text = doc.page_content
            sentences = sent_tokenize(text)

            current_chunk, char_count = [], 0

            for sent in sentences:
                if char_count + len(sent) <= self.chunks_size:
                    current_chunk.append(sent)
                    char_count += len(sent)
                else:
                    chunks.append(" ".join(current_chunk))
                    metadata.append({"source_file": doc.metadata.get("source_file", f"doc_{idx}")})

                    # Overlap handling
                    overlap_sents = current_chunk[-self.overlap//50:]
                    current_chunk = overlap_sents + [sent]
                    char_count = sum(len(s) for s in current_chunk)
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                metadata.append({"source_file": doc.metadata.get("source_file", f"doc_{idx}")})
        
        # Ensure processed_dir exists
        self.processed_dir.mkdir(parents = True, exist_ok = True)

        # Save the chunks and metadata
        with open(self.chunks_file, "wb") as f1, open(self.metadata_file, "wb") as f2:
            pickle.dump(chunks, f1)
            pickle.dump(metadata, f2)
            self.logger.info(f"Saved {len(chunks)} chunks and metadata to {self.processed_dir}")
        
        return chunks, metadata