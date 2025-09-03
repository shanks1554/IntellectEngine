# test_root.py
from dotenv import load_dotenv
from src.config import embedding_model
from src.vectorstore import VectorStore
from src.rag import RAG

# Load environment variables
load_dotenv(override=True)

def main():
    # Initialize VectorStore (will auto-build FAISS if needed)
    vs = VectorStore(embedding_model=embedding_model, embedding_dim=768)

    # Initialize RAG system
    rag_system = RAG(vector_store=vs, k=1)

    # Test query
    query = "Explain supervised learning in simple terms."
    answer = rag_system.answer_query(query)
    
    print("\n--- Query ---")
    print(query)
    print("\n--- Answer ---")
    print(answer)

if __name__ == "__main__":
    main()
