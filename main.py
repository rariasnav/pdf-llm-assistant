import os
from utils.pdf_loader import extract_text_from_pdf
from utils.text_splitter import split_text
from utils.embedding_store import create_embeddings ,load_vectorstore
from utils.question_answering import build_qa_chain, ask_question
from dotenv import load_dotenv

load_dotenv()

PDF_FOLDER = "data/pdfs"
INDEX_FOLDER = "vectorstore"

# list all pdfs in the PDF_FOLDER
def list_pdfs():
    return [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

# verify if the index exists for the given pdf
# if not, create it and return the path for the index
def ensure_index_for_pdf(pdf_filename):
    name = os.path.splitext(pdf_filename)[0]
    index_path = os.path.join(INDEX_FOLDER, f"{name}_index")
    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
    
    if not os.path.exists(index_path):
        print(f"Processing {pdf_filename}...")
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text)
        create_embeddings(chunks, save_path=index_path)
    else:
        print(f"Index already exists for {pdf_filename}")
    return index_path


# shows the user the available pdfs and asks for a selection
# or asks for all documents
def select_mode():
    print("\n Available PDFs:")
    pdf_files = list_pdfs()
    for i, name in enumerate(pdf_files, start=1):
        print(f"{i}. {name}")
    print(f"{len(pdf_files)+1}. All Documents")
    
    choice = input("\n Select a document to query (number): ").strip()
    if not choice.isdigit():
        return None, None
    
    idx = int(choice)
    if 1 <= idx <= len(pdf_files):
        return [ensure_index_for_pdf(pdf_files[idx - 1])], [pdf_files[idx - 1]]
    elif idx == len(pdf_files) + 1:
        index_paths = []
        names = []
        for f in pdf_files:
            index_paths.append(ensure_index_for_pdf(f))
            names.append(f)
        return index_paths, names
    return None, None

# combines multiple FAISS vectorstores into one
def combine_vectorstores(paths):
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings()
    stores = [load_vectorstore(path, embeddings) for path in paths]
    combined = stores[0]
    for store in stores[1:]:
        combined.merge_from(store)
    return combined

def main():
    index_paths, names = select_mode()
    if not index_paths:
        print("Invalid selection. Exiting.")
        return
    
    print(f"\n Loaded index for: {', '.join(names)}")
    
    vectorstore = combine_vectorstores(index_paths)
    qa = build_qa_chain(vectorstore)
    
    while True:
        question = input("\n Ask a question (or 'exit' to quit): ")
        if question.lower() in ['exit', 'quit']:
            break        
        response = ask_question(question, qa)
        print(f"\n Response: {response}")
        
if __name__ == "__main__":
    main()