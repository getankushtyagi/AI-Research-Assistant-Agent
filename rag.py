# here we stire the web content into the vector DB

from tools import search_tool
from scraper import read_webpage
from vectore_store import vector_store


def collect_research(query):
    response = search_tool.invoke(query) # here we use tavily to search the result from the internet and give 5 result as we mention in the fucntion
    
    # Get the results list from the response
    result = response.get('results', [])
    
    documents=[]
    
    for r in result:
        url= r.get("url") #here we get all the website urls
        text = read_webpage(url) # here with the help of scraper we scrap the data 
        
        if text:
            documents.append(text)
    if documents:
        vector_store.add_texts(documents)
        
    return len(documents)


# you can test the integartion while running the file test_rag.py