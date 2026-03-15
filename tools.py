from langchain_tavily import TavilySearch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

search_tool = TavilySearch( # here we are creating a search tool with the help of tavily and we are giving the query and the number of results we want to get from the internet
    max_results=5
)