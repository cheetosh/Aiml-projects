import streamlit as st
import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Streamlit App Title
st.title("📚 RAG-based Knowledge Assistant (Offline Mode)")

# File upload and query input
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
query = st.text_input("Ask a question about the document:")

if uploaded_file is not None and query:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and process PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Split & embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    # ✅ Local LLM pipeline (no API or internet needed after first download)
    summarizer = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=summarizer)

    # Build RAG
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    # Get answer
    result = qa.run(query)

    # Display result
    st.subheader("Answer:")
    st.write(result)
