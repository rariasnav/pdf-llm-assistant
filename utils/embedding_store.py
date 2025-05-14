from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


def create_embeddings(chunks, save_path="vectorstore/faiss_index"):
    # wrap chunks as Document objects (needed by langchain)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # load emebeddings model (OpenAI)
    embeddings = OpenAIEmbeddings()
    
    # create FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # save localy
    vectorstore.save_local(save_path)
    
    return vectorstore

def load_vectorstore(path="vectorstore/faiss_index", embeddings=None):
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore