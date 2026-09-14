import os
from crewai import Agent, Task, Crew, Process, LLM

from tools.pdf_tool import PDFSearchTool
from tools.search_tool import SerpAPISearchTool
from tools.weather_tool import WeatherTool

llm = LLM(model="gpt-4o", temperature=0.2)

pdf_tool = PDFSearchTool()
search_tool = SerpAPISearchTool()
weather_tool = WeatherTool()

# --- Worker Agents ---

pdf_agent = Agent(
    role="PDF Research Analyst",
    goal="Answer user questions strictly using content retrieved from the uploaded PDF.",
    backstory=(
        "You are an expert document analyst. You only answer using facts found "
        "in the PDF Search Tool's output. If nothing relevant is found, say so clearly."
    ),
    tools=[pdf_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

search_agent = Agent(
    role="Web Research Specialist",
    goal="Find accurate, up-to-date information from the web to answer user questions.",
    backstory=(
        "You are skilled at crafting effective search queries and synthesizing "
        "web search results into clear, cited answers."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

weather_agent = Agent(
    role="Weather Analyst",
    goal="Provide accurate current weather information for any requested location.",
    backstory="You specialize in interpreting weather API data into human-friendly summaries.",
    tools=[weather_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# --- Manager Agent (used automatically by Process.hierarchical) ---

manager_agent = Agent(
    role="Task Manager",
    goal=(
        "Understand the user's request and delegate it to the correct specialist agent: "
        "PDF Research Analyst for document questions, Web Research Specialist for general/"
        "current-events questions, or Weather Analyst for weather queries. Combine results "
        "into one final clear answer."
    ),
    backstory="You are an experienced coordinator who routes work efficiently and never does the specialist work yourself.",
    llm=llm,
    verbose=True,
    allow_delegation=True,
)


def build_crew(user_query: str) -> Crew:
    task = Task(
        description=(
            f"Answer the following user query as accurately as possible: '{user_query}'. "
            "Decide which specialist(s) are needed: PDF content, web search, or weather. "
            "If the query needs multiple sources, delegate to multiple agents and combine results."
        ),
        expected_output="A clear, well-structured final answer to the user's query, citing the source (PDF/Web/Weather) used.",
        agent=manager_agent,
    )

    crew = Crew(
        agents=[pdf_agent, search_agent, weather_agent],
        tasks=[task],
        manager_agent=manager_agent,
        process=Process.hierarchical,
        verbose=True,
    )
    return crew


def run_query(user_query: str) -> str:
    crew = build_crew(user_query)
    result = crew.kickoff()
    return str(result)