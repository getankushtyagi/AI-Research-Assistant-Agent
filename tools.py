from langchain_tavily import TavilySearch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

search_tool = TavilySearch(
    max_results=5
)