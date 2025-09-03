# main.py
from dotenv import load_dotenv
import gradio as gr

from src.config import embedding_model
from src.vectorstore import VectorStore
from src.rag import RAG
from src.logger import get_logger

# Initialize logger
logger = get_logger("main")

def init_system():
    logger.info("Loading environment variables...")
    load_dotenv(override=True)

    logger.info("Initializing VectorStore...")
    vs = VectorStore(embedding_model=embedding_model, embedding_dim=768)

    logger.info("Initializing RAG system...")
    rag = RAG(vector_store=vs, k=3)

    logger.info("System initialized successfully.")
    return rag

rag_system = init_system()

# Gradio function
def ask_engine(query: str):
    if not query.strip():
        logger.warning("Empty query received.")
        return "⚠ Please enter a valid question."

    logger.info(f"Received query: {query}")
    return rag_system.answer_query(query)

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; font-size: 24px; font-weight: bold;">
            🧠 IntellectEngine - RAG Assistant
        </div>
        <div style="text-align: center; font-size: 14px; color: gray; margin-bottom: 15px;">
            Ask any ML/DL related question, and let IntellectEngine find the answer.
        </div>
        """
    )

    with gr.Row():
        query_box = gr.Textbox(
            label="Enter your question",
            placeholder="Type your query here...",
            lines=2,
            scale=8
        )
        submit_btn = gr.Button("🔍 Submit", variant="primary", scale=2)

    output_box = gr.Markdown(
        value="💡 Ready! Ask me something...",
        elem_id="output-box"
    )

    # Show "Processing..." while waiting for response
    submit_btn.click(
        lambda x: "⏳ Processing your query...",
        inputs=query_box,
        outputs=output_box
    ).then(
        ask_engine,
        inputs=query_box,
        outputs=output_box
    )

if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    demo.launch()