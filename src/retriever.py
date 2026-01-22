from langchain_pinecone import PineconeVectorStore
from src.config import PINECONE_INDEX_NAME, PINECONE_NAMESPACE
from src.embeddings import get_embedder

def get_retriever(k=5):
    embedder = get_embedder()
    vs = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedder,
        namespace=PINECONE_NAMESPACE,  # ✅ IMPORTANT
    )
    return vs.as_retriever(search_kwargs={"k": k})
