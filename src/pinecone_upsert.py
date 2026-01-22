import os
import uuid
from pinecone import Pinecone

from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE
from src.embeddings import get_embedder

def upsert_chunks(chunks, batch_size=100):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    embedder = get_embedder()

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        texts = [d.page_content for d in batch]
        vectors = embedder.embed_documents(texts)  # list of 384-dim vectors

        upserts = []
        for doc, vec in zip(batch, vectors):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)
            doc_id = f"{os.path.basename(str(src))}_p{page}_{uuid.uuid4().hex[:10]}"

            metadata = {
                "text": doc.page_content,     # ✅ store full chunk text
                "source": str(src),
                "page": page,
                "page_label": doc.metadata.get("page_label"),
            }
            upserts.append((doc_id, vec, metadata))

        index.upsert(vectors=upserts, namespace=PINECONE_NAMESPACE)
        print(f"✅ Upserted {min(start + batch_size, total)}/{total}")

    print("🎉 Done: Upsert complete.")
