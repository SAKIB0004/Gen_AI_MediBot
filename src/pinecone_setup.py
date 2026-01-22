from pinecone import Pinecone, ServerlessSpec
from src.config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION
)

DIMENSION = 384

def ensure_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = {i["name"] for i in pc.list_indexes()}
    if PINECONE_INDEX_NAME in existing:
        print("ℹ️ Index already exists:", PINECONE_INDEX_NAME)
        return

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    print("✅ Created index:", PINECONE_INDEX_NAME)
