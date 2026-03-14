import requests
from bs4 import BeautifulSoup #this is used to scrap the data from the internet 


def read_webpage(url):
    try:
        response=requests.get(url,timeout=10) # download web page url
        soup = BeautifulSoup(response.text , "html.parser") # parse HTML structure
        
        paragraph= soup.find_all("p") # here we extract only p tag i.e paragraph
        text="".join(p.text for p in paragraph) 
        return text[:5000]
    except Exception as e:
        return None
