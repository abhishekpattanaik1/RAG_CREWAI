================================================================================
 MULTI-AGENT RAG ASSISTANT (CrewAI + Streamlit)
 PDF Question Answering | Web Search | Weather | Manager-Delegated Routing
================================================================================

--------------------------------------------------------------------------------
1. OVERVIEW
--------------------------------------------------------------------------------

This project is a multi-agent system built with CrewAI that answers user
questions by routing them to the correct specialist agent:

  - PDF Research Analyst  -> answers questions using a RAG pipeline over an
                              uploaded PDF (Chroma vector store)
  - Web Research Specialist -> answers questions using live web search
                                (SerpAPI)
  - Weather Analyst        -> answers questions using live weather data
                                (OpenWeather API)

A Manager Agent sits above these three and decides which specialist(s) a
given query needs, delegates the work, and combines the results into one
final answer. Routing is handled natively by CrewAI's hierarchical process
(Process.hierarchical) -- no manual if/else routing logic is required.

A Streamlit UI provides PDF upload/ingestion and a chat-style question box.

--------------------------------------------------------------------------------
2. PROJECT STRUCTURE
--------------------------------------------------------------------------------

rag_crew_app/
|-- .venv/                 (virtual environment -- do not name this "rag")
|-- app.py                 Streamlit entrypoint
|-- crew.py                Agents, Manager, Crew, and task orchestration
|-- tools/
|   |-- __init__.py
|   |-- pdf_tool.py        PDFSearchTool   -> RAG retrieval over the PDF
|   |-- search_tool.py     SerpAPISearchTool -> web search
|   |-- weather_tool.py    WeatherTool     -> live weather lookup
|-- rag/
|   |-- __init__.py
|   |-- ingest.py          PDF -> chunks -> embeddings -> vector store
|   |-- vectorstore.py     VectorStoreManager (create/load/query Chroma)
|-- chroma_db/              persisted vector store (created after first ingest)
|-- uploads/                 uploaded PDFs are saved here
|-- requirements.txt
|-- .env                    API keys (not committed to version control)
|-- README.txt               this file

--------------------------------------------------------------------------------
3. AGENT INTEGRATION -- HOW IT FITS TOGETHER
--------------------------------------------------------------------------------

3.1 Tools (tools/)
--------------------
Each tool is a CrewAI BaseTool subclass with a `name`, a `description` (used
by the LLM to decide when to call it), and a `_run()` method that does the
actual work:

  PDFSearchTool     - queries the Chroma vector store built from the
                        ingested PDF and returns the most relevant chunks.
  SerpAPISearchTool - sends a query to SerpAPI (Google Search API) and
                        returns the top organic results.
  WeatherTool        - calls the OpenWeather "current weather" endpoint for
                        a given city and returns a formatted summary.

3.2 Worker Agents (crew.py)
------------------------------
  pdf_agent      -> role: PDF Research Analyst, uses PDFSearchTool only.
  search_agent   -> role: Web Research Specialist, uses SerpAPISearchTool only.
  weather_agent  -> role: Weather Analyst, uses WeatherTool only.

  Each worker has allow_delegation=False -- workers execute their own tool,
  they do not hand work off to anyone else.

3.3 Manager Agent
--------------------
  manager_agent has allow_delegation=True and no tools of its own. It reads
  the incoming Task description, decides which worker(s) are relevant, and
  delegates. This is what "Manager Agent for task delegation" means in
  practice under CrewAI's hierarchical process.

3.4 Crew & Process
----------------------
  Crew(
      agents=[pdf_agent, search_agent, weather_agent],
      tasks=[task],
      manager_agent=manager_agent,
      process=Process.hierarchical,
  )

  Process.hierarchical is what activates manager-driven delegation. The
  manager_agent is NOT included in the `agents=[...]` list passed for
  worker execution -- CrewAI treats it as a separate coordination role.

3.5 LLM Backend
------------------
  All agents share one LLM instance, created via CrewAI's own LLM class
  (NOT langchain_openai.ChatOpenAI -- recent CrewAI versions validate the
  `llm` field as a string or crewai.LLM instance, so a raw LangChain chat
  model object will fail Pydantic validation):

      from crewai import LLM
      llm = LLM(model="gpt-4o", temperature=0.2)

  crewai.LLM routes calls through LiteLLM under the hood and reads
  OPENAI_API_KEY from the environment at construction time.

3.6 Streamlit UI (app.py)
-----------------------------
  - Sidebar: upload a PDF -> "Ingest PDF" button -> builds/updates the
    Chroma vector store via rag/ingest.py.
  - Main panel: free-text question box -> "Run" button -> calls
    crew.run_query(query), which builds and kicks off the Crew, and
    displays the manager's final combined answer.

--------------------------------------------------------------------------------
4. SETUP
--------------------------------------------------------------------------------

4.1 Create and activate a virtual environment
-------------------------------------------------
  IMPORTANT: do not name the venv "rag" -- that collides with the local
  "rag" package (rag/ingest.py, rag/vectorstore.py) and causes
  ModuleNotFoundError. Use ".venv" instead.

    python -m venv .venv
    source .venv/bin/activate        (macOS/Linux)
    .venv\Scripts\activate           (Windows)

4.2 Install dependencies
----------------------------
    pip install -r requirements.txt

4.3 Configure API keys
--------------------------
  Create a .env file in the project root:

    OPENAI_API_KEY=your_openai_key
    SERPAPI_API_KEY=your_serpapi_key
    OPENWEATHER_API_KEY=your_openweather_key

4.4 Run the app
-------------------
  Run from the project root (not from inside a subfolder):

    streamlit run app.py

--------------------------------------------------------------------------------
5. USAGE
--------------------------------------------------------------------------------

  1. Open the Streamlit app in your browser (usually http://localhost:8501).
  2. In the sidebar, upload a PDF and click "Ingest PDF" to build the
     vector store.
  3. Type a question in the main panel and click "Run":
       - PDF questions:      "Summarize section 2 of the document"
       - Web questions:      "What's the latest news on <topic>?"
       - Weather questions:  "What's the weather in Hyderabad right now?"
       - Combined questions: the manager can delegate to more than one
         specialist and merge the results.

--------------------------------------------------------------------------------
6. KNOWN ISSUES / TROUBLESHOOTING LOG (from this project's build process)
--------------------------------------------------------------------------------

  a) ModuleNotFoundError: No module named 'rag'
     Cause: virtual environment folder was also named "rag", shadowing the
     actual rag/ package.
     Fix: rename the venv to ".venv"; ensure rag/__init__.py exists; run
     streamlit from the project root.

  b) ModuleNotFoundError: No module named 'langchain.text_splitter'
     ModuleNotFoundError: No module named 'langchain.schema'
     Cause: LangChain 0.1+ split the monolithic `langchain` package into
     langchain-core, langchain-community, langchain-text-splitters,
     langchain-openai, langchain-chroma, etc. Old top-level imports were
     removed, not just deprecated.
     Fix: use the new import paths, e.g.
       from langchain_text_splitters import RecursiveCharacterTextSplitter
       from langchain_core.documents import Document
       from langchain_openai import OpenAIEmbeddings, ChatOpenAI
       from langchain_chroma import Chroma
       from langchain_community.document_loaders import PyPDFLoader
     See requirements.txt for a pinned, mutually compatible version set.

  c) ImportError: cannot import name 'PDFSearchTool' / 'WeatherTool'
     Cause: the target file either had the wrong class pasted into it, was
     empty, or did not exist yet (e.g. tools/weather_tool.py was missing).
     Fix: verify each tools/*.py file defines the class matching its
     filename:
       grep -n "^class " tools/*.py
     and recreate any missing file with the correct class definition.

  d) pydantic_core.ValidationError on Agent(llm=...)
       "Input should be a valid string ... Input should be a valid
        dictionary or instance of BaseLLM"
     Cause: recent CrewAI versions require `llm` to be either a model-name
     string or an instance of crewai.LLM -- a raw langchain_openai.ChatOpenAI
     object no longer validates.
     Fix:
       from crewai import LLM
       llm = LLM(model="gpt-4o", temperature=0.2)
     and pass this `llm` object to every Agent(...) call.

  e) "You have no credits remaining" on kickoff()
     Cause: the OpenAI (or other LLM provider) account behind
     OPENAI_API_KEY has no billing credit / free quota left. This is a
     billing/account-level issue, not a code bug -- CrewAI's LLM call to
     the provider is being rejected by the provider itself.
     Fix options:
       - Add billing/credits to the OpenAI account tied to OPENAI_API_KEY.
       - Switch to a provider/model with available quota, e.g. point
         crewai.LLM at a different provider (Anthropic, a local Ollama
         model, Groq, etc.) by changing the model string and the
         corresponding API key env var, e.g.:
             llm = LLM(model="ollama/llama3.1", base_url="http://localhost:11434")
         or
             llm = LLM(model="anthropic/claude-sonnet-4-6")
         (set ANTHROPIC_API_KEY accordingly).
       - Check usage/limits on the provider's dashboard before re-running.

--------------------------------------------------------------------------------
7. NOTES FOR FUTURE WORK
--------------------------------------------------------------------------------

  - Routing reliability: hierarchical delegation is LLM-driven and can
    misroute ambiguous queries. A deterministic pre-classifier or explicit
    Process.sequential branching is an alternative if this matters.
  - Vector store persistence: currently reads/writes Chroma from disk on
    every query; consider caching in st.session_state for lower latency.
  - Cost/latency: hierarchical process = manager reasoning + delegation +
    worker execution = more LLM calls than a single-agent pipeline.
  - Multi-PDF support: VectorStoreManager already accepts a
    collection_name, so per-document collections are possible instead of
    one shared store.
  - Package/version drift: CrewAI and LangChain both evolve their public
    APIs frequently. If a new ModuleNotFoundError or ValidationError
    appears after an upgrade, check each package's current changelog
    before assuming the project code is wrong.

================================================================================
 END OF README
================================================================================
