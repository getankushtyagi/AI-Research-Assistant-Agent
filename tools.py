from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain.tools import tool
from vectore_store import vector_store
from scraper import read_webpage
# Load environment variables from .env file
load_dotenv()

search_tool = TavilySearch( # here we are creating a search tool with the help of tavily and we are giving the query and the number of results we want to get from the internet
    max_results=5
)

@tool
def retrieve_docs(query: str) -> str:
    """Search stored research knowledge from vector database."""
    docs=vector_store.similarity_search(query,k=3)
    results="\n\n".join(d.page_content for d in docs)
    return results

@tool
def read_page(url: str) -> str:
    """Read the content of a webpage given its URL."""
    return read_webpage(url)

@tool
def store_research(text: str) -> str:
    """store research content into the vector database"""
    
    vector_store.add_texts([text])
    return "Research stored successfully"
    

tools=[
    search_tool,
    read_page,
    retrieve_docs,
    store_research,
]