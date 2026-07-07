# RAG-Based Document Question Answering System 🤖📄

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload PDF documents, ask questions based on the content, and receive accurate, document-specific answers. It combines the power of **Google Gemini** for language processing, **HuggingFace** for embeddings, **ChromaDB** for efficient vector storage and retrieval, and **Streamlit** for a user-friendly interface.

---

## Features ✨

* **PDF Processing**: Extracts text from uploaded PDF documents and splits it into manageable chunks for embedding and storage.
* **Hybrid Search Retrieval**: Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) for vector similarity search (ChromaDB) combined with **BM25 Keyword Search** for highly accurate context retrieval.
* **Question Answering**: Leverages Google Gemini's advanced language models to generate accurate responses by retrieving and analyzing relevant document chunks.
* **Interactive Interface**: Provides a simple and intuitive interface using Streamlit for uploading documents, entering queries, and viewing results.

---

## How to Use 🚀

Follow the steps below to run and interact with the project:

### 1. Clone the Repository
Clone the repository to your local system using the following command:
```bash
git clone https://github.com/Harshit045/RAG_Document_Question_Answering-.git
cd RAG_Document_Question_Answering-
```

### 2. Create and Activate a Virtual Environment 🐍
Create a virtual environment and activate it to isolate project dependencies:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies 📦
Install the required Python libraries using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Obtain API Keys 🔑
Get your API keys for:
* **Gemini**: Sign up at [Google AI Studio](https://aistudio.google.com/) to obtain an API key. 
> **Note:** This key will be entered securely via the Streamlit interface when running the app. No keys are stored in the code!

### 5. Run the Application 🏃
Launch the Streamlit application:
```bash
streamlit run streamlit_app.py
```

### 6. Access the Application 🌐
Once the application is running, open your browser and navigate to the URL provided by Streamlit, typically [http://localhost:8501](http://localhost:8501).

### 7. Upload a Document 📄
Use the left sidebar interface to upload one or more PDF files containing the content you want to query, and click "Process Documents".

### 8. Ask Questions ❓
* Enter your Gemini API Key in the sidebar.
* Enter your question in the main query box. The chatbot will:
  * Retrieve relevant chunks of text from the uploaded document using Hybrid Search.
  * Generate a precise and context-aware response using Gemini.

---

## Project Structure 📁

```text
├── streamlit_app.py    # Main application file with Streamlit interface
├── rag_engine.py       # Handles PDF processing, embedding, and hybrid retrieval
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```
