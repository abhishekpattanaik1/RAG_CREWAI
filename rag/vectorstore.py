import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class VectorStoreManager:
    """
    Manages creation, persistence, loading, and querying of the
    Chroma vector store used by the PDF RAG tool.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "pdf_documents",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self._store: Optional[Chroma] = None

    def create(self, documents: List[Document]) -> Chroma:
        """Build a fresh vector store from a list of chunked documents."""
        os.makedirs(self.persist_dir, exist_ok=True)
        self._store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )
        self._store.persist()
        return self._store

    def load(self) -> Chroma:
        """Load an existing persisted vector store from disk."""
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(
                f"No vector store found at '{self.persist_dir}'. "
                "Ingest a PDF first via VectorStoreManager.create()."
            )
        self._store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        return self._store

    def get_or_load(self) -> Chroma:
        """Return the in-memory store if already loaded, else load from disk."""
        if self._store is not None:
            return self._store
        return self.load()

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to an existing store (e.g. a second PDF upload)."""
        store = self.get_or_load()
        store.add_documents(documents)
        store.persist()

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Run a similarity search and return the top-k matching chunks."""
        store = self.get_or_load()
        return store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4):
        """Same as above but also returns relevance/distance scores."""
        store = self.get_or_load()
        return store.similarity_search_with_score(query, k=k)

    def as_retriever(self, k: int = 4):
        """Expose a LangChain retriever interface, useful for chains elsewhere."""
        store = self.get_or_load()
        return store.as_retriever(search_kwargs={"k": k})

    def collection_exists(self) -> bool:
        """Check whether a persisted collection already exists on disk."""
        return os.path.exists(self.persist_dir) and len(os.listdir(self.persist_dir)) > 0