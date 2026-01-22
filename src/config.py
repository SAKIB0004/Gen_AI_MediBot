import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medibot-medical")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "medical")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

PDF_DIR = os.getenv("PDF_DIR", "data/medical_books")

def validate_env():
    missing = []
    if not PINECONE_API_KEY: missing.append("PINECONE_API_KEY")
    if not GROQ_API_KEY: missing.append("GROQ_API_KEY")
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")
