import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_hybrid_retriever(uploaded_files):
    """
    Processes uploaded files, creates an in-memory vector store and BM25 index,
    and returns a function that can be used to query both.
    """
    documents = []
    
    # Save uploaded files temporarily to read them with PyPDFLoader
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())

    if not documents:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # 1. Create In-Memory Chroma DB (No persist_directory means it won't lock files)
    vectorstore = Chroma.from_documents(chunks, get_embeddings())

    # 2. Create In-Memory BM25 Index
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. Define the Hybrid Search function
    def hybrid_search(query, k=4):
        # Vector Search
        vector_results = vectorstore.similarity_search_with_score(query, k=k)
        
        # BM25 Keyword Search
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
        
        bm25_results = []
        for i in top_n:
            if bm25_scores[i] > 0:
                bm25_results.append((chunks[i], bm25_scores[i]))

        # Combine results
        combined_chunks = {}
        for doc, score in vector_results:
            combined_chunks[doc.page_content] = doc
        for doc, score in bm25_results:
            combined_chunks[doc.page_content] = doc

        return list(combined_chunks.values())

    return hybrid_search
