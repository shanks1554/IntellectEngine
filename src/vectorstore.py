import faiss
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from src.preprocessor import Preprocessor
from src.logger import get_logger

preprocessor = Preprocessor()
class VectorStore:
    def __init__(self, embedding_model, embedding_dim=768, faiss_dir = 'data/faiss_index', preprocessor = preprocessor):
        """
        Initialize VectorStore.
        Will auto-load existing FAISS DB if available, otherwise requires building.
        """
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.faiss_dir = Path(faiss_dir)
        self.preprocessor = preprocessor
        # Files
        self.index_file = self.faiss_dir/ "intellect_engine.index"
        self.chunks_file = self.faiss_dir/ "chunks.pkl"
        self.metadata_file = self.faiss_dir/ "metadata.pkl"
        self.valid_indices_file = self.faiss_dir/ "valid_indices.pkl"
        self.index_info_file = self.faiss_dir/ "index_info.json"

        # Runtime variable
        self.index = None
        self.chunks = []
        self.metadata = []
        self.valid_indices = []

        self.logger = get_logger(self.__class__.__name__)

        # Ensure directory exists
        self.faiss_dir.mkdir(parents = True, exist_ok = True)

        # Try auto-load
        if self.load():
            self.logger.info("✅ Using existing FAISS database")
        else:
            self.logger.info("⚠️ No FAISS database found, building it with .build()")
            chunks, metadata = self.preprocessor.loader_method()
            self.build(chunks=chunks, metadata=metadata)
    
    def generate_embeddings(self, chunks, batch_size=50):
        """Generate embeddings for chunks in batches."""
        all_embeddings = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        self.logger.info(f"Generating embeddings for {len(chunks)} chunks in {total_batches} batches")

        for i in tqdm(range(0, len(chunks), batch_size), desc = "Embedding batches"):
            batch_chunks = chunks[i:i + batch_size]

            try:
                batch_embeddings = self.embedding_model.embed_documents(batch_chunks)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                self.logger.error(f"Error in batch {i // batch_size + 1}: {e}")
                all_embeddings.extend([None] * len(batch_chunks))
        
        self.logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def create_index(self, embeddings, chunks, metadata):
        """Create a FAISS index from embeddings"""
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        self.valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]

        self.logger.info(f"Creating FAISS index with {len(valid_embeddings)} valid embeddings")

        index = faiss.IndexFlatIP(self.embedding_dim)
        embeddings_array = np.array(valid_embeddings).astype("float32")
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)

        self.index = index
        self.chunks = chunks
        self.metadata = metadata

        self.logger.info(f"FAISS index created with {index.ntotal} vectors")
    
    def save(self):
        """Save FAISS index and supporting data"""
        faiss.write_index(self.index, str(self.index_file))

        with open(self.chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(self.valid_indices_file, "wb") as f:
            pickle.dump(self.valid_indices, f)
        
        index_info = {
            "total_chunks": len(self.chunks),
            "valid_chunks": len(self.valid_indices),
            "embedding_dimension": self.index.d,
            "index_type": "IndexFlatIP",
            "created_at": datetime.now().isoformat(),
        }
        with open(self.index_info_file, "w") as f:
            json.dump(index_info, f, indent = 2)
        
        self.logger.info(f"FAISS database saved to {self.faiss_dir}")
    
    def load(self):
        """Load FAISS index and supporting data if available"""
        if all(f.exists() for f in [self.index_file, self.chunks_file, self.metadata_file, self.valid_indices_file]):
            self.index = faiss.read_index(str(self.index_file))
            with open(self.chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            with open(self.valid_indices_file, "rb") as f:
                self.valid_indices = pickle.load(f)
            
            self.logger.info(f"Loaded FAISS database from {self.faiss_dir}")
            return True
        else:
            return False
    
    def build(self, chunks, metadata, batch_size = 50):
        """Build a new FAISS index from chunks + metadata and save it."""
        embeddings = self.generate_embeddings(chunks, batch_size = batch_size)
        self.create_index(embeddings, chunks, metadata)
        self.save()

    def query(self, query, k = 1):
        """Query FAISS index with natural language input."""
        if self.index is None:
            raise ValueError("FAISS index not loaded or created")
        
        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.valid_indices):
                original_idx = self.valid_indices[idx]
                chunk = self.chunks[original_idx]
                meta = self.metadata[original_idx]

                results.append({
                    'chunk': chunk,
                    'metadata': meta,
                    "score": float(score)
                })
        
        return results

