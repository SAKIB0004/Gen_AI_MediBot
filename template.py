import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s :  - %(levelname)s : ] : %(message)s : ')


list_of_files = [
    "src/__init__.py",
    "src/config.py",
    "src/pdf_loader.py",
    "src/chunking.py",
    "src/embeddings.py",
    "src/pinecone_setup.py",
    "src/pinecone_upsert.py",
    "src/retriever.py",
    "src/rag_groq.py",
    
    
    "data/medical_books/",
    "assets/styles.css",

    "ingest.py",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for file: {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists and is not empty: {filepath}")

logging.info("File and directory setup complete.")
# This script creates a predefined set of directories and files for a project structure.
