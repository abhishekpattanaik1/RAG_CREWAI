from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.vectorstore import VectorStoreManager

def build_vectorstore(pdf_path: str, persist_dir: str = "./chroma_db"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    manager = VectorStoreManager(persist_dir=persist_dir)
    if manager.collection_exists():
        manager.add_documents(chunks)   # append if PDF already ingested before
    else:
        manager.create(chunks)
    return manager
