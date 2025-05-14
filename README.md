# 🧠 PDF LLM Assistant

An intelligent assistant that allows you to upload and query multiple PDFs using natural language, powered by OpenAI and LangChain. Built with Python and Streamlit.

> Example: "Summarize this document in 10 lines as if I were 15 years old" ✅

---

## 🚀 Features

- Upload one or more PDFs via web interface.
- Automatically processes documents and stores semantic vector indexes using FAISS.
- Asks questions about **one or all documents at once**.
- Uses OpenAI LLMs to provide context-aware answers.
- Full terminal version and web version available.
- Clean architecture for future extensions (e.g. model swapping, file export, chat logs).

---

## 🛠️ Tech Stack

- Python 3.10+
- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Streamlit](https://streamlit.io/)

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/rariasnav/pdf-llm-assistant.git
cd pdf-llm-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Then add your OpenAI API key to .env
```
---

## 🧪 Run the App

```bash
# Web UI (Streamlit)
streamlit run app.py

### Terminal Mode
python main.py
```

---

📁 Folder Structure
```bash
.
├── app.py                  # Web interface
├── main.py                 # Terminal mode
├── data/pdfs/              # Folder for uploaded PDFs
├── vectorstore/            # FAISS vector indexes
├── utils/                  # Modular functions (PDF, embeddings, QA)
├── .env.example
├── requirements.txt
└── README.md
```

---

👤 Author
Rafael Arias – [LinkedIn](https://www.linkedin.com/in/rafael-arias-navarro/)
Engineer | AI Developer | 🇨🇴 Seeking opportunities with relocation in Europe or UK
