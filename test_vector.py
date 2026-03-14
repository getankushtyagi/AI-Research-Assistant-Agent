from vectore_store import vector_store

docs=[
    "AI agents are becoming very popular",
    "LangChain helps build LLM applications",
    "Vector databases store embeddings"
]

vector_store.add_texts(docs) # this will basiclly add text into vector db using chroma and embedding 

result=vector_store.similarity_search("ai agents") # now this function query the vector DB and find the result 

for r in result: # here we print all the result one by one 
    print(r.page_content)