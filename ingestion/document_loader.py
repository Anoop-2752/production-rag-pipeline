import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from config import DATA_DIR


def load_pdf(file_path: str):
    """Load a single PDF file."""
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f"✅ Loaded: {file_path} — {len(documents)} pages")
    return documents


def load_all_pdfs(directory: str = DATA_DIR):
    """Load all PDFs from the data directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Created directory: {directory}")

    loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"✅ Total pages loaded: {len(documents)}")
    return documents