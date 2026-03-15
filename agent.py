import os 

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from vectore_store import vector_store
from langchain.agents import create_agent
from tools import tools

load_dotenv()
# here we create the gemini with the hep of langchain google function 
llm = ChatGoogleGenerativeAI( 
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2 #Temperature controls creativity vs accuracy.
)



def retrieve_context(query):
    docs=vector_store.similarity_search(query,k=3)
    context="\n\n".join([d.page_content for d in docs])
    return context

def answer_question(query):
    context=retrieve_context(query)
    
    prompt = f"""
    You are an AI research assistant.

    Use the context below to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer clearly and cite important points.
    """
    response=llm.invoke(prompt)
    return response.content


# to test the complete pipeline run the rest_agent.py file 

# Create the agent with the new LangGraph-based pattern
agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are an AI research assistant. 
    Use the available tools to search for information and answer questions.
    When using the vector knowledge search, look for relevant stored research.
    Be thorough and cite your sources.""",
    debug=True
)

def run_agent(query):
    result = agent_executor.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result
