import streamlit as st
import os
from rag_backend import create_hybrid_retriever
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="RAG Document Q&A", page_icon="🤖")

st.title("🤖 RAG-Based Document Question Answering System")
st.markdown("Upload your PDF documents and ask questions based on their content. The system uses **Hybrid Search** (Vector + BM25) and **Gemini** to provide accurate answers.")

# Sidebar for inputs
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password", help="Get this from Google AI Studio")
    
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents 🚀"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    # Create the retriever purely in memory (No file locking issues!)
                    retriever = create_hybrid_retriever(uploaded_files)
                    st.session_state['retriever'] = retriever
                    st.session_state['ready'] = True
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
        else:
            st.warning("Please upload at least one PDF file.")

# Main area for Q&A
st.header("💬 Ask a Question")
query = st.text_input("Enter your question about the documents:")

if st.button("Get Answer"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar first!")
        st.stop()
        
    if not st.session_state.get('ready', False):
        st.warning("Please upload and process documents first using the sidebar.")
        st.stop()
            
    if query:
        with st.spinner("Searching for answers..."):
            try:
                # Use the in-memory retriever
                hybrid_search = st.session_state['retriever']
                docs = hybrid_search(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                if not context:
                    st.info("I couldn't find relevant information in the documents.")
                else:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0,
                        google_api_key=api_key
                    )
                    
                    prompt = f"""
                    You are a helpful assistant. Use the following retrieved context to answer the user's question.
                    If you don't know the answer based on the context, just say that you don't know. 
                    Do not make up information.
                    
                    Context:
                    {context}
                    
                    Question:
                    {query}
                    
                    Answer:
                    """
                    
                    response = llm.invoke(prompt)
                    st.write("### Answer:")
                    st.write(response.content)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
