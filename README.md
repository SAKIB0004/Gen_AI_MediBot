# MediBot

> **A source-grounded medical book question-answering system built with RAG, Pinecone, Groq, and Streamlit.**

## Overview

MediBot is a Retrieval-Augmented Generation (RAG) application that answers natural-language questions using content retrieved from a medical textbook. Instead of generating free-form medical claims, the system grounds responses in retrieved book chunks and applies safety-oriented guardrails to reduce hallucinations.

This project is designed as a portfolio-ready applied GenAI system that demonstrates document ingestion, vector indexing, retrieval, LLM-based answer generation, chat-style UX, and deployment readiness.

## Why This Project Matters

Large language models can sound convincing even when they are wrong. In medical and educational domains, that risk is especially important. MediBot addresses this by combining:

- **document-grounded retrieval** from a trusted source,
- **LLM reasoning** only over retrieved context,
- **score-based fallback behavior** when context is weak,
- **clear safety disclaimers** for informational use.

The result is a practical example of how to build a safer, more reliable question-answering system for knowledge-intensive domains.

## Key Features

- **Medical book-only answering** using retrieved textbook chunks
- **RAG pipeline** powered by Pinecone + LangChain + Groq
- **Low-hallucination guardrails** with similarity threshold checks
- **Conversation-aware UX** with chat history support
- **Modern Streamlit interface** with custom styling and source display
- **Source expanders** for transparent answer traceability
- **Namespace-safe retrieval** for predictable vector search behavior
- **PDF ingestion pipeline** for chunking, embedding, and upserting content
- **Dockerized deployment support**
- **AWS EC2 + ECR + GitHub Actions CI/CD compatibility**

## Tech Stack

### Core Technologies

- **Language:** Python
- **Frontend/UI:** Streamlit
- **Framework:** LangChain
- **Vector Database:** Pinecone
- **LLM Provider:** Groq
- **Embeddings:** Configurable embedding backend
- **Containerization:** Docker
- **Cloud / Deployment:** AWS EC2, Amazon ECR, GitHub Actions

### Supporting Components

- PDF loading and preprocessing
- Semantic chunking pipeline
- Vector upsert utilities
- Environment-based configuration management
- Streamlit chat state and memory-aware retrieval flow

## Architecture / Workflow

```text
Medical PDF
   │
   ▼
PDF Loader
   │
   ▼
Chunking Pipeline
   │
   ▼
Embedding Generation
   │
   ▼
Pinecone Vector Index
   │
   ▼
Retriever
   │
   ▼
Groq-powered RAG Chain
   │
   ▼
Streamlit Chat Interface
```

### Inference Flow

1. A user asks a question in the Streamlit interface.
2. The app retrieves the most relevant chunks from Pinecone.
3. A similarity threshold check is applied.
4. If retrieved evidence is weak, the system returns an explicit fallback such as *"I don't know based on the provided book."*
5. If relevant evidence exists, the Groq-backed RAG chain generates a response grounded in retrieved context.
6. The answer is displayed with optional source references.

## System Design Notes

This project is structured as a modular RAG system with clear separation of concerns:

- **`app.py`** handles the Streamlit UI, chat interaction, and response rendering.
- **`ingest.py`** runs the ingestion pipeline for PDF-to-vector conversion.
- **`src/`** contains reusable modules for configuration, loading, chunking, embeddings, vector storage, retrieval, and RAG generation.
- **`.streamlit/`** provides Streamlit runtime configuration.
- **`.github/workflows/`** supports CI/CD automation.

## Dataset

This project uses a **medical textbook PDF** as its knowledge source.

- The source file is expected locally at:

```text
Data/medical_books/Medical_Book.pdf
```
- You can replace the PDF with another textbook or domain-specific reference file if needed.

## Model Details

This project does not train a custom deep learning model. Instead, it uses an applied GenAI pipeline composed of:

- **Retriever:** Pinecone vector search
- **Embedder:** Configurable embedding backend
- **Generator:** Groq LLM via LangChain
- **Response policy:** retrieval-first, source-grounded answer generation

### Training Process

Not applicable in the classical ML sense.

Instead of model training, the project performs:

- document loading,
- chunking,
- embedding generation,
- vector indexing,
- retrieval-time answer generation.

## Evaluation and Reliability Approach

This project focuses on practical reliability rather than benchmark training metrics.

### Current Reliability Measures

- Retrieved-context-only answering
- Explicit fallback on weak retrieval
- Source display for transparency
- Safety disclaimer for educational use only
- Namespace-safe vector retrieval


## Project Structure

```text
Gen_AI_MediBot/
├── .github/
│   └── workflows/              # CI/CD workflows
├── .streamlit/                 # Streamlit configuration
├── Data/                       # Local medical PDF input
│           
├── assets/                     # UI assets / styling resources
├── research/                   # Research notebooks / experiments / drafts
├── src/
│   ├── chunking.py             # Text chunking logic
│   ├── config.py               # Environment loading and constants
│   ├── embeddings.py           # Embedding model setup
│   ├── pdf_loader.py           # PDF reading utilities
│   ├── pinecone_setup.py       # Pinecone index setup
│   ├── pinecone_upsert.py      # Vector upsert pipeline
│   ├── rag_groq.py             # Groq-based RAG chain
│   └── retriever.py            # Retriever utilities
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── app.py                      # Streamlit application entrypoint
├── ingest.py                   # Ingestion runner
├── requirements.txt
├── setup.py
└── template.py
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SAKIB0004/Gen_AI_MediBot.git
cd Gen_AI_MediBot
```

### 2. Create a Virtual Environment

#### Using `venv`

```bash
python -m venv .venv
source .venv/bin/activate
```

#### On Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root based on `.env.example`.

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_NAMESPACE=medical
GROQ_API_KEY=your_groq_api_key
```

> Add any additional configuration variables required by your current embedding setup.

## Setup Instructions

### 1. Add the PDF Knowledge Source

Place your medical textbook PDF here:

```text
Data/medical_books/Medical_Book.pdf
```

### 2. Ingest the PDF into Pinecone

```bash
python ingest.py
```

This step:

- loads the PDF,
- splits it into chunks,
- generates embeddings,
- pushes chunk vectors into Pinecone.

## Run Instructions

### Local Streamlit Run

```bash
streamlit run app.py
```

### Recommended Streamlit Network Binding

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then open:

```text
http://localhost:8501
```

## Docker Usage

### Build the Image

```bash
docker build -t medibot .
```

### Run the Container

```bash
docker run -d \
  --name medibot \
  -e PINECONE_API_KEY="your_pinecone_api_key" \
  -e GROQ_API_KEY="your_groq_api_key" \
  -e PINECONE_INDEX_NAME="your_index_name" \
  -e PINECONE_NAMESPACE="medical" \
  -p 8501:8501 \
  medibot
```

Open the application at:

```text
http://localhost:8501
```

## Deployment

This project is designed to support containerized deployment on AWS using a lightweight CI/CD workflow.

### Deployment Stack

- **Amazon ECR** for container image storage
- **AWS EC2** for application hosting
- **GitHub Actions** for build and deployment automation
- **Docker** for packaging and runtime consistency

### Deployment Workflow

1. Code is pushed to the `main` branch.
2. GitHub Actions builds the latest Docker image.
3. The image is pushed to **Amazon ECR**.
4. A **self-hosted GitHub Actions runner** on the EC2 instance pulls the latest image.
5. Docker stops the old container, starts the new one, and serves the Streamlit app on port `8501`.
6. Runtime secrets are injected securely from GitHub Actions into the container environment.

### Runtime Access

After successful deployment, the application can be accessed through the EC2 instance public IP or a custom domain.

```text
http://<EC2_PUBLIC_IP>:8501
```

> For a stable public endpoint, use an **Elastic IP** or connect a domain name through a reverse proxy such as Nginx.


## Secrets Management

This project requires API credentials and deployment secrets, but **real secret values should never be committed to the repository**.

### Recommended Secret Handling

Use **GitHub Actions Secrets** for CI/CD and environment variable injection.

Typical secrets for this project include:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `ECR_REPO`
- `PINECONE_API_KEY`
- `GROQ_API_KEY`

These secrets should be stored in:

- **GitHub Repository Secrets**, or
- **GitHub Environment Secrets** for deployment-specific environments


## Example Questions

You can test the application with prompts like:

- `What is hypertension?`
- `What are the symptoms of diabetes?`
- `Explain anemia in simple words.`
- `What is the value of RBC count and is it in range?` *(if the uploaded source contains lab-related content)*

## Example Output / Results

**Example answer style:**

- grounded in retrieved context,
- concise and readable,
- accompanied by sources when available,
- falls back safely if relevant evidence is missing.

```text
Question:
What is hypertension?

Answer:
Hypertension refers to persistently elevated blood pressure. Based on the provided source,
it is associated with increased cardiovascular risk and may require lifestyle modification
or medical management depending on severity.
```

> Replace this section with actual screenshots or real sample outputs from your running app.


### Demo Links

- **Live Demo:** `http://13.206.80.6:8501`

## Safety and Limitations

- This application is intended for **educational and informational use only**.
- It is **not a medical diagnosis or treatment system**.
- Response quality depends on:
  - the quality of the source PDF,
  - the chunking strategy,
  - the embedding model,
  - retriever relevance.
- If the answer is not present in the uploaded book, the system should prefer a safe fallback.


## License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.


