import os
from crewai.tools import BaseTool
from serpapi import GoogleSearch

class SerpAPISearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = (
        "Searches the web for current information, news, or facts "
        "not available in the PDF. Input should be a search query string."
    )

    def _run(self, query: str) -> str:
        params = {
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": 5
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic = results.get("organic_results", [])
        if not organic:
            return "No web results found."
        formatted = "\n\n".join(
            f"{r.get('title')}\n{r.get('snippet', '')}\n{r.get('link')}"
            for r in organic[:5]
        )
        return formatted