import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from config import DATA_DIR


def load_pdf(file_path: str):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {file_path} ({len(documents)} pages)")
    return documents


def load_all_pdfs(directory: str = DATA_DIR):
    """
    Recursively loads all PDFs under `directory`.
    Creates the directory if it doesn't exist yet — useful on first run.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created data directory at {directory}")

    loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages across all PDFs")
    return documents
