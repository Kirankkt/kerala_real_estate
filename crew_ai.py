# crew_ai.py
from crewai import Agent
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import os


def initialize_agents(openai_key, serper_key):
    """Initialize agents with provided API keys."""
    # Set up environment variables
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["SERPER_API_KEY"] = serper_key

    # Parameters for LLM
    parameters = {"temperature": 0.2, "max_tokens": 300}
    llm = ChatOpenAI(model="gpt-3.5-turbo", params=parameters)

    # Tools
    search_tool = SerperDevTool(api_key=os.environ["SERPER_API_KEY"])

    # Agents
    real_estate_agent = Agent(
        llm=llm,
        role="Real Estate Researcher",
        goal="Find properties with water views in Trivandrum.",
        tools=[search_tool],
        verbose=1,
    )
    furniture_agent = Agent(
        llm=llm,
        role="Furniture Storyteller",
        goal="Create stories for Kerala-style furniture targeting NRIs.",
        verbose=1,
    )
    website_agent = Agent(
        llm=llm,
        role="Website Design Consultant",
        goal="Analyze design elements for luxury real estate and furniture websites.",
        tools=[search_tool],
        verbose=1,
    )

    return {
        "real_estate_agent": real_estate_agent,
        "furniture_agent": furniture_agent,
        "website_agent": website_agent,
    }
