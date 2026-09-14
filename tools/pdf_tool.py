# tools/pdf_tool.py
from crewai.tools import BaseTool
from rag.vectorstore import VectorStoreManager

class PDFSearchTool(BaseTool):
    name: str = "PDF Search Tool"
    description: str = (
        "Searches the uploaded PDF's content to answer questions. "
        "Input should be a natural language question about the document."
    )

    def _run(self, query: str) -> str:
        manager = VectorStoreManager()
        if not manager.collection_exists():
            return "No PDF has been ingested yet. Please upload a PDF first."
        results = manager.similarity_search(query, k=4)
        if not results:
            return "No relevant content found in the PDF."
        return "\n\n".join(
            f"[Page {d.metadata.get('page', '?')}]: {d.page_content}" for d in results
        )
