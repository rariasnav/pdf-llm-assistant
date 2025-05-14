from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


load_dotenv()

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(
        temperature=0.2,
        model="gpt-3.5-turbo"
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

def ask_question(question, chain):
    result = chain.invoke({"query": question})
    return result['result']