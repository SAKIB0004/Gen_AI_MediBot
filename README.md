# 🩺 MediBot – Medical Book Question Answering System

MediBot is a **Retrieval-Augmented Generation (RAG)** based medical book question answering application. It allows users to ask natural language questions and receive **source-grounded answers strictly from a medical textbook**, minimizing hallucinations and improving reliability.

This project demonstrates an **end-to-end GenAI pipeline** using **LangChain, Pinecone, Groq LLMs, and Streamlit**, with a production-style UI and strong safety controls.

---

## ✨ Key Features

- 📚 **Medical Book–Only Answers**  
  Answers are generated *only* from retrieved medical book chunks.

- 🔎 **RAG Pipeline (Retrieval-Augmented Generation)**  
  Combines semantic search (Pinecone) with LLM reasoning (Groq).

- 🧠 **Low Hallucination Guard**  
  Similarity-score gating ensures the bot says *"I don't know"* when context is weak.

- 💬 **ChatGPT-like Experience**  
  - Typing indicator ("Thinking…")  
  - Progressive answer streaming (character-by-character)

- 🩺 **Medical Safety Disclaimer**  
  Clear separation between informational content and medical advice.

- 🎨 **Modern, Creative UI**  
  Custom CSS, H2 header, badges, source expanders, and clean chat layout.

---

## 🧱 Tech Stack

| Layer | Technology |
|-----|-----------|
| Frontend | Streamlit |
| LLM | Groq (via LangChain) |
| Embeddings | HuggingFace / OpenAI (configurable) |
| Vector DB | Pinecone |
| Framework | LangChain |
| Language | Python |

---

## 📂 Project Structure

```
GEN_AI_MediBot/
│
├── app.py                  # Streamlit application (UI + RAG logic)
├── ingest.py               # End-to-end ingestion runner
├── requirements.txt
├── README.md
│
├── Data/
│   └── medical_books/
│       └── Medical_Book.pdf   # (local file – not committed to GitHub)
│
├── src/
│   ├── config.py           # Env loading & constants
│   ├── pdf_loader.py       # PDF reading
│   ├── chunking.py         # Text chunking logic
│   ├── embeddings.py       # Embedding model
│   ├── pinecone_setup.py   # Pinecone index creation
│   ├── pinecone_upsert.py  # Vector upsert logic
│   ├── retriever.py        # Namespace-safe retriever
│   └── rag_groq.py         # RAG chain with Groq
│
└── assets/
    └── styles.css          # UI styling
```

---

## 🔄 RAG Architecture (High Level)

1. **PDF Ingestion**  
   Medical book is loaded and split into semantically meaningful chunks.

2. **Embedding Generation**  
   Each chunk is converted into vector embeddings.

3. **Vector Storage**  
   Embeddings + full text are stored in **Pinecone (namespace: medical)**.

4. **Query Time Flow**  
   - User asks a question  
   - Relevant chunks retrieved from Pinecone  
   - Similarity threshold check applied  
   - Groq LLM generates an answer *only from retrieved context*

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SAKIB0004/Gen_AI_MediBot.git
cd Gen_AI_MediBot
```

### 2️⃣ Create & Activate Environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables

Create a `.env` file in the project root (this file is **not committed**):

```
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=medibot-medical
PINECONE_NAMESPACE=medical
GROQ_API_KEY=your_groq_key
```

### 4️⃣ Add Medical Book PDF (Required)

Place your medical textbook PDF at:

```
Data/medical_books/Medical_Book.pdf
```

> The PDF is intentionally **excluded from the GitHub repository** to keep the repo lightweight and avoid copyright issues.

### 5️⃣ Ingest the Medical Book

```bash
python ingest.py
```

> This step loads the PDF, chunks it, generates embeddings, and upserts vectors into Pinecone.

### 6️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🛡️ Safety & Reliability

- ✅ Answers strictly limited to retrieved book context
- ✅ Namespace-safe retrieval (no silent empty searches)
- ✅ Similarity-score gating
- ✅ Explicit medical disclaimer

This makes MediBot suitable for **educational and informational use only**, not diagnosis or treatment.

---

## 📌 Example Use Cases

- Medical students revising concepts
- Educational demonstrations of RAG systems
- GenAI portfolio project
- Interview-ready applied AI system

---

## 🧪 Future Enhancements

- Token-level streaming from Groq
- Confidence score visualization
- Multiple books / multi-namespace support
- Feedback loop (👍 / 👎)
- Deployment on Hugging Face Spaces or AWS

---

## 👨‍💻 Author

Built as part of a **Generative AI learning journey with LangChain & RAG systems**.

If you're reviewing this as a recruiter or mentor: this project demonstrates **end-to-end GenAI system design**, not just prompt usage.

---

⭐ *If you found this project useful, consider starring the repository!*

