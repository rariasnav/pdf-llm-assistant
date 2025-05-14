import os
import streamlit as st
from dotenv import load_dotenv
from utils.pdf_loader import extract_text_from_pdf
from utils.text_splitter import split_text
from utils.embedding_store import create_embeddings, load_vectorstore
from utils.question_answering import build_qa_chain, ask_question
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PDF_FOLDER = "data/pdfs"
INDEX_FOLDER = "vectorstore"

def list_pdfs():
    return [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

def ensure_index_for_pdf(pdf_filename):
    name = os.path.splitext(pdf_filename)[0]
    index_path = os.path.join(INDEX_FOLDER, f"{name}_index")
    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
    
    if not os.path.exists(index_path):
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text)
        create_embeddings(chunks, save_path=index_path)
    return index_path

def combine_vectorstores(paths):   
    embeddings = OpenAIEmbeddings()
    stores = [load_vectorstore(path, embeddings) for path in paths]
    combined = stores[0]
    for store in stores[1:]:
        combined.merge_from(store)
    return combined

# streamlit app
st.set_page_config(page_title="PDF LLM Assistant", layout="centered")
st.title("PDF LLM Assistant")

# 1. Upload PDF files
st.sidebar.header("Upload your PDFs")
uploaded_files = st.sidebar.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

# 2. Process selected files
if uploaded_files:
    for file in uploaded_files:
        save_path = os.path.join(PDF_FOLDER, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())
    st.sidebar.success("Files uploaded successfully. You can now select them below.")
        
pdf_files = list_pdfs()
selected_files = st.multiselect("Select PDF(s) to query", options=pdf_files)       
        
if selected_files:
    with st.spinner("Processing documents..."):
        index_paths = [ensure_index_for_pdf(f) for f in selected_files]
        vs = combine_vectorstores(index_paths)
        qa = build_qa_chain(vs)
        
    question = st.text_input("Ask a question about the selected PDFs:")
    if question:
        with st.spinner("Generating response..."):
            response = ask_question(question, qa)
            st.markdown("### 💬 Response:")
            st.write(response)