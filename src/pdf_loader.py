from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_pdfs(folder_path: str):
    loader = DirectoryLoader(
        path=folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    return loader.load()
