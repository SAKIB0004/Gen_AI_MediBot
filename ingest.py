from src.config import validate_env, PDF_DIR
from src.pinecone_setup import ensure_index
from src.pdf_loader import load_pdfs
from src.chunking import chunk_documents
from src.pinecone_upsert import upsert_chunks

def main():
    validate_env()
    ensure_index()

    docs = load_pdfs(PDF_DIR)
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=150)

    print("Total pages loaded:", len(docs))
    print("Total chunks created:", len(chunks))

    upsert_chunks(chunks, batch_size=100)

if __name__ == "__main__":
    main()
