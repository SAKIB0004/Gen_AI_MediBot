from langchain_huggingface import HuggingFaceEmbeddings

def get_embedder():
    # all-MiniLM-L6-v2 => 384-dim
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
