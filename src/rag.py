import google.generativeai as genai
from vectorstore import VectorStore
from logger import get_logger
import os
from dotenv import load_dotenv

load_dotenv(override = True)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key = GEMINI_API_KEY)


class RAG:
    def __init__(self, vector_store: VectorStore, k: int = 1):
        """
        Initialize the RAG system.
        : param vector_store: Instance of VectorStore
        : param k: Number of top chunks to retrieve (set k=1 for single best answer)
        """

        self.vector_store = vector_store
        self.k = k
        self.logger = get_logger(self.__class__.__name__)

        # Initialize Gemini model
        self.model = genai.GenerativeModel(model_name='gemini-2.5-pro', )
        self.logger.info("✅ Gemini model initialized for RAG responses.")
    
    def create_rag_prompt(self, query: str, retrieved_chunks: list) -> str:
        """
        Create a learner-friendly prompt for Gemini.
        : param query: User Query
        : param retrieved_chunks: List of retrieved chunks from VectorStore
        : return: Formatted prompt string
        """
        combined_context = "\n\n".join([r['chunk'] for r in retrieved_chunks])

        prompt = f"""You are IntellectEngine, an AI assistant specialized in AI, Machine Learning, and Deep Learning.

        Based on the following context, explain the concept clearly and understandably for a learner.

        CONTEXT:
        {combined_context}

        QUESTION: {query}
        Instructions:
        - Provide a concise, human-like explanation.
        - Paraphrase and simplify concepts for easy understanding.
        - Structure your answer logically.
        - Focus on teaching and clarity, not citations.

        ANSWER:
        """
        return prompt

    def answer_query(self, query: str) -> str:
        """
        Process a query end-to-end: retrieve relevant chunks + generate response.
        : param query: User query
        : return: Response text from Gemini
        """
        self.logger.info(f"Processing query: {query}")

        # Step 1: Retrieve most relevant chunks from FAISS
        retrieved = self.vector_store.query(query, k = self.k)
        if not retrieved:
            self.logger.info("⚠️ No relavent chunk found.")
            return "Sorry, I couldn't find relevant information."
        
        # Step 2: Create RAG prompt
        prompt = self.create_rag_prompt(query, retrieved)

        # Step 3: generated response using Gemini
        try:
            response = self.model.generate_content(prompt)
            self.logger.info("✅ Response generated successfully.")
            return response.text

        except Exception as e:
            self.logger.error(f"❌ Error generating response: {e}")
            return "Sorry, I couldn't generate a response at this time."