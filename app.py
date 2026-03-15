from fastapi import FastAPI
from agent import run_agent



app=FastAPI()

@app.get("/ask")
def ask(q:str):
    answer = run_agent(q)
    
    return {
        "response":answer
    }