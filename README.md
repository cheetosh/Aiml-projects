📚 RAG-based Knowledge Assistant
This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, Hugging Face embeddings, and FAISS.

🔑 Features
Upload PDF documents and ask natural language questions.
Uses embeddings + vector search to retrieve context.
Answers generated with LLM + document chunks.
⚙️ Tech Stack
LangChain
Hugging Face (MiniLM embeddings)
FAISS
OpenAI API
Streamlit UI
🚀 How to Run
pip install -r requirements.txt
streamlit run rag_app.py

---

### 🔹 README for **NER Pipeline**
```markdown
# 📝 Named Entity Recognition (NER) Pipeline

This project extracts **structured entities** (dates, organizations, locations) from unstructured text using SpaCy.

## 🔑 Features
- Input any text and extract entities.
- Supports PERSON, ORG, DATE, GPE, MONEY, etc.

## ⚙️ Tech Stack
- SpaCy
- Streamlit UI

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run ner_app.py

---

### 🔹 README for **Technical Report Summarizer**
```markdown
# 📄 Technical Report Summarizer

This project summarizes long technical reports using Hugging Face **BART** transformer.

## 🔑 Features
- Paste or upload long text.
- Get an abstractive summary in seconds.

## ⚙️ Tech Stack
- Hugging Face Transformers
- Streamlit UI

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run summarizer_app.py
