import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Initialize the Gemini embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model = 'models/embedding-001',
    google_api_key = GEMINI_API_KEY,
    output_dimensionality = 768
)

# Test embedding generation to get dimension
try:
    test_vector = embedding_model.embed_query("Test Sentence")
    embedding_dim = len(test_vector)
    print(f"Gemini embeddings initialized. Dimension: {embedding_dim}")
except Exception as e:
    embedding_model = None
    embedding_dim = None
    print(f"❌ Failed ro initialize Gemini embeddings: {e}")

# Export variables for external use
__all__ = ['embedding_model', 'embedding_dim']