## 🧠 IntellectEngine - RAG Application

IntellectEngine is a **Retrieval-Augmented Generation (RAG)** application that combines vector search with LLMs to provide accurate, context-aware answers from a collection of machine learning and deep learning resources.

It uses **LangChain**, **FAISS**, and **Google Generative AI** for embeddings & retrieval, with a **Gradio-based interface** for user interaction.

## 📂 Project Structure

```base
IntellectEngine/
│
├── 📁config
│   └── rag_config.json
│
├── 📁data
│   ├── 📁raw            # Raw PDF files
│   ├── 📁processed      # Preprocessed documents (chunks + metadata)
│   └── 📁faiss_index    # FAISS vectorstore
│
├── 📁logs               # Application logs
├── 📁src
│   ├── logger.py        # Logging utility
│   ├── data_loader.py   # Raw data handling
│   ├── preprocessor.py  # Document chunking/cleaning
│   ├── vectorstore.py   # FAISS vectorstore handling
│   ├── rag.py           # RAG pipeline
│   └── config.py        # Embedding model configuration
│
├── main.py              # Gradio interface (entry point)
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

## ⚙️ Installation

1. **Clone the Repository**

    ```base
    git clone https://github.com/shanks1554/IntellectEngine.git
    cd IntellectEngine
    ```

2. **Create virtual environment**

    ```base
    python -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```base
    pip install -r requirements.txt
    ```

4. **Set environment variables**

    Create a .env file in the root directory and add you GEMINI API KEY:

    ```base
    GEMINI_API_KEY = your_api_key
    ```

## ▶️ Usage

Run the application

```base
python main.py
```

This will launch a Gradio web interface where you can enter queries.


## 🔍 Features

- 📖 Load and preprocess raw PDFs into a knowledge base

- 🗂 Build / load FAISS vectorstore (skips recomputation if already exists)

- 🤖 Retrieval-Augmented Generation (RAG) pipeline

- 🎛 Gradio interface with submit button and loading screen

- 📜 Logging of system events and queries

## 🛠️ Tech Stack

- LangChain – Orchestration framework

- FAISS – Vector similarity search

- Google Generative AI (Gemini) – Embeddings + LLM

- Gradio – Frontend interface

- NLTK & SentenceTransformers – Text preprocessing & embeddings

- PyPDF – PDF text extraction

## 📌 Future Improvements

- Add citations (show sources with answers)

- Multi-turn chat interface with memory

- Advanced UI (dark mode, sidebar for documents, etc.)