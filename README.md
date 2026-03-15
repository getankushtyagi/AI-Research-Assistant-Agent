# AI Research Assistant (Multi-Tool RAG Agent)

A sophisticated AI-powered research assistant that combines web search, web scraping, vector database storage, and intelligent question-answering capabilities. This system uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware responses based on both stored knowledge and real-time web data.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Function Reference](#function-reference)
- [Setup Instructions](#setup-instructions)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)

---

## 🎯 Overview

This AI Research Assistant leverages multiple technologies to create an intelligent system that can:
- **Search the web** using Tavily API for real-time information
- **Scrape and process** web content from URLs
- **Store knowledge** in a vector database for efficient retrieval
- **Answer questions** using Google's Gemini AI with contextual understanding
- **Operate as an agent** with multiple tools for complex research tasks
- **Serve via API** using FastAPI for web integration

### Key Technologies
- **LangChain**: Framework for building LLM-powered applications
- **Google Gemini 2.5 Flash**: Large Language Model for intelligent responses
- **ChromaDB**: Vector database for semantic search
- **HuggingFace Embeddings**: Text-to-vector transformation
- **Tavily Search**: Web search API
- **BeautifulSoup**: Web scraping
- **FastAPI**: REST API framework

---

## 🏗️ Architecture

```
User Query
    ↓
┌─────────────────────────────────────┐
│         FastAPI Endpoint            │
│          (app.py)                   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│       Agent Orchestrator            │
│       (agent.py)                    │
│   - LangGraph-based agent           │
│   - Tool selection & execution      │
└─────────────────────────────────────┘
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
┌──────────┐      ┌──────────────┐
│  Tools   │      │ RAG Pipeline │
│(tools.py)│      │  (rag.py)    │
└──────────┘      └──────────────┘
    │                   │
    ├─→ Tavily Search   ├─→ Web Scraper
    ├─→ Vector Search   │   (scraper.py)
    └─→ Web Reader      │
                        ↓
              ┌──────────────────┐
              │ Vector Database  │
              │ (vectore_store.py)│
              │   - ChromaDB     │
              │   - Embeddings   │
              └──────────────────┘
```

---

## 📁 Project Structure

```
AI Research Assistant/
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Python dependencies
├── app.py                 # FastAPI web server
├── agent.py               # Main agent logic and LLM integration
├── tools.py               # Tool definitions for the agent
├── rag.py                 # RAG pipeline for web research
├── vectore_store.py       # Vector database configuration
├── scraper.py             # Web scraping utilities
├── test_vector.py         # Test script for vector operations
├── test_rag.py            # Test script for RAG pipeline
├── test_agent.py          # Test script for agent Q&A
└── data/                  # ChromaDB persistent storage (auto-created)
```

---

## 🔧 Core Components

### 1. **vectore_store.py** - Vector Database Setup

**Purpose**: Creates and configures a persistent vector database for storing and retrieving document embeddings.

**Why we use it**: 
- Enables semantic search over stored documents
- Converts text into mathematical vectors (embeddings) that capture meaning
- Allows finding similar content based on context, not just keywords
- Provides persistent storage so knowledge is retained between sessions

**Key Components**:

#### `embedding_model` (HuggingFaceEmbeddings)
```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu', 'trust_remote_code': True},
    encode_kwargs={'normalize_embeddings': True}
)
```
- **What it does**: Converts text into 384-dimensional vectors
- **Why we use it**: 
  - Captures semantic meaning of text (similar concepts have similar vectors)
  - Lightweight model (runs on CPU) suitable for production
  - Fast inference for real-time applications
  - Normalized embeddings improve similarity search accuracy

#### `vector_store` (Chroma)
```python
vector_store = Chroma(
    collection_name="research_agent",
    embedding_function=embedding_model,
    persist_directory='./data'
)
```
- **What it does**: Creates a ChromaDB instance with persistent storage
- **Why we use it**:
  - Stores document vectors with metadata
  - Enables fast similarity search using HNSW index
  - Persists data to disk (./data directory)
  - Automatically handles vector indexing and retrieval
  - Supports filtering and metadata-based queries

---

### 2. **scraper.py** - Web Content Extraction

**Purpose**: Downloads and extracts text content from web pages.

**Why we use it**:
- Converts HTML pages into clean, usable text
- Filters out navigation, ads, and formatting
- Provides raw content for vector storage
- Handles errors gracefully (timeouts, invalid URLs)

#### `read_webpage(url)` Function

```python
def read_webpage(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraph = soup.find_all("p")
        text = "".join(p.text for p in paragraph)
        return text[:5000]
    except Exception as e:
        return None
```

**Parameters**:
- `url` (str): The webpage URL to scrape

**Returns**:
- First 5000 characters of text content, or None if error

**Why each step**:
1. **`requests.get(url, timeout=10)`**:
   - Downloads HTML content from the URL
   - Timeout prevents hanging on slow servers
   - Returns HTTP response object

2. **`BeautifulSoup(response.text, "html.parser")`**:
   - Parses HTML into a navigable tree structure
   - "html.parser" is Python's built-in parser (no extra dependencies)
   - Allows easy extraction of specific HTML elements

3. **`soup.find_all("p")`**:
   - Extracts all paragraph tags (`<p>`)
   - Paragraphs contain the main text content
   - Skips headers, navigation, scripts, etc.

4. **`"".join(p.text for p in paragraph)`**:
   - Combines all paragraph text into one string
   - Removes HTML tags, keeping only text
   - Creates a continuous document for processing

5. **`return text[:5000]`**:
   - Limits to 5000 characters to avoid memory issues
   - Sufficient for most article content
   - Prevents token limits in LLM processing

6. **`except Exception: return None`**:
   - Catches network errors, invalid URLs, parsing failures
   - Graceful degradation (system continues if one page fails)
   - Prevents crashes from malformed HTML

---

### 3. **rag.py** - Research Collection Pipeline

**Purpose**: Orchestrates the complete RAG (Retrieval-Augmented Generation) pipeline for collecting and storing web research.

**Why we use it**:
- Automates the research workflow: search → scrape → store
- Builds knowledge base from web sources
- Enables the agent to access up-to-date information
- Combines multiple data sources into unified storage

#### `collect_research(query)` Function

```python
def collect_research(query):
    response = search_tool.invoke(query)
    result = response.get('results', [])
    documents = []
    
    for r in result:
        url = r.get("url")
        text = read_webpage(url)
        if text:
            documents.append(text)
    
    if documents:
        vector_store.add_texts(documents)
    
    return len(documents)
```

**Parameters**:
- `query` (str): Search query to research

**Returns**:
- Number of documents successfully stored

**Step-by-step explanation**:

1. **`search_tool.invoke(query)`**:
   - Calls Tavily API to search the web
   - Returns top 5 relevant URLs with metadata
   - Uses AI-powered search ranking
   - **Why**: Gets authoritative, relevant sources for the query

2. **`response.get('results', [])`**:
   - Extracts the list of search results
   - Default to empty list if no results
   - **Why**: Safely handles API response structure

3. **Loop through results**:
   ```python
   for r in result:
       url = r.get("url")
   ```
   - Iterates through each search result
   - Extracts the URL from each result
   - **Why**: Process each source individually

4. **`text = read_webpage(url)`**:
   - Scrapes content from the URL
   - Returns cleaned text or None
   - **Why**: Converts web pages to text for storage

5. **`if text: documents.append(text)`**:
   - Only adds successfully scraped content
   - Filters out failed scrapes
   - **Why**: Ensures only valid data enters the database

6. **`vector_store.add_texts(documents)`**:
   - Converts texts to embeddings
   - Stores in ChromaDB with automatic indexing
   - **Why**: Makes content searchable via semantic similarity

7. **`return len(documents)`**:
   - Reports how many documents were stored
   - **Why**: Provides feedback on research success rate

---

### 4. **tools.py** - Agent Tool Definitions

**Purpose**: Defines the tools that the AI agent can use to accomplish tasks.

**Why we use it**:
- Gives the agent capabilities beyond just text generation
- Enables interaction with external systems (search, database, web)
- Provides structured, reliable functions for the agent
- Separates concerns (tools vs. agent logic)

#### Environment Setup

```python
load_dotenv()
```
- **What it does**: Loads API keys from .env file
- **Why**: Keeps secrets out of code, enables easy configuration

#### `search_tool` (TavilySearch)

```python
search_tool = TavilySearch(max_results=5)
```
- **What it does**: Creates a web search tool using Tavily API
- **Why we use it**:
  - Provides real-time web search capability
  - Returns AI-ranked, relevant results
  - Includes URL, title, and content snippets
  - More reliable than Google scraping
  - Built-in rate limiting and error handling

#### `retrieve_docs(query)` Tool

```python
@tool
def retrieve_docs(query: str) -> str:
    """Search stored research knowledge from vector database."""
    docs = vector_store.similarity_search(query, k=3)
    results = "\n\n".join(d.page_content for d in docs)
    return results
```

**What it does**: Searches the vector database for relevant stored documents

**Why each part**:

1. **`@tool` decorator**:
   - Registers function as a LangChain tool
   - Auto-generates tool description from docstring
   - Enables agent to discover and use this tool
   - **Why**: Makes the function usable by the agent

2. **`vector_store.similarity_search(query, k=3)`**:
   - Converts query to embedding vector
   - Finds the 3 most similar documents
   - Uses cosine similarity for matching
   - **Why k=3**: Balances context quality vs. token limits

3. **`"\n\n".join(d.page_content for d in docs)`**:
   - Combines multiple documents into one string
   - Separates with double newlines for readability
   - **Why**: Creates coherent context for the LLM

**Use cases**:
- Answering questions from previously stored research
- Finding relevant background information
- Accessing internal knowledge base

#### `read_page(url)` Tool

```python
@tool
def read_page(url: str) -> str:
    """Read the content of a webpage given its URL."""
    return read_webpage(url)
```

**What it does**: Scrapes and returns content from a specific URL

**Why we use it**:
- Allows agent to read specific pages on demand
- Useful for following up on search results
- Provides detailed content beyond search snippets
- **Why separate from retrieve_docs**: Different use cases (web vs. database)

#### `tools` List

```python
tools = [search_tool, read_page, retrieve_docs]
```
- **What it does**: Bundles all tools for the agent
- **Why**: Single point of configuration for agent capabilities

---

### 5. **agent.py** - AI Agent and Question Answering

**Purpose**: Core intelligence layer that combines LLM reasoning with tool usage.

**Why we use it**:
- Provides natural language understanding
- Orchestrates tool usage based on user intent
- Generates human-like responses
- Maintains conversation context

#### Environment and LLM Setup

```python
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)
```

**Why each parameter**:

1. **`model="gemini-2.5-flash"`**:
   - Latest Google Gemini model (as of 2026)
   - "Flash" variant: optimized for speed and cost
   - Strong reasoning and multilingual capabilities
   - **Why**: Balance between performance and cost

2. **`temperature=0.2`**:
   - Controls randomness in responses (0.0 = deterministic, 2.0 = creative)
   - Low temperature = more factual, consistent answers
   - **Why 0.2**: Research applications need accuracy over creativity

#### `retrieve_context(query)` Function

```python
def retrieve_context(query):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    return context
```

**Parameters**:
- `query` (str): User's question

**Returns**:
- Combined text from 3 most relevant documents

**Why we use it**:
- Finds relevant background information
- Reduces hallucination by grounding responses in facts
- Provides citations and specific details
- **Why k=3**: Optimal balance of context vs. token usage

#### `answer_question(query)` Function

```python
def answer_question(query):
    context = retrieve_context(query)
    
    prompt = f"""
    You are an AI research assistant.
    Use the context below to answer the question.
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer clearly and cite important points.
    """
    
    response = llm.invoke(prompt)
    return response.content
```

**Parameters**:
- `query` (str): User's question

**Returns**:
- AI-generated answer based on stored context

**Why this pattern**:

1. **Retrieval-first approach**:
   - Gets relevant context before asking LLM
   - Ensures answer is based on factual data
   - **Why**: RAG pattern reduces hallucination

2. **Structured prompt**:
   - Clear role definition ("AI research assistant")
   - Explicit instructions ("use context", "cite points")
   - Separated sections for clarity
   - **Why**: Improves response quality and consistency

3. **`llm.invoke(prompt)`**:
   - Sends prompt to Gemini API
   - Returns structured response object
   - **Why invoke vs. generate**: Synchronous, simpler for API use

4. **`response.content`**:
   - Extracts text from response object
   - **Why**: Returns clean string, not metadata

**Use case**: Direct Q&A without agent overhead (faster, simpler)

#### Agent Creation

```python
agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are an AI research assistant. 
    Use the available tools to search for information and answer questions.
    When using the vector knowledge search, look for relevant stored research.
    Be thorough and cite your sources.""",
    debug=True
)
```

**What it does**: Creates a LangGraph-based agent that can use tools

**Why each parameter**:

1. **`model=llm`**:
   - The LLM that powers agent reasoning
   - Makes decisions about which tools to use
   - **Why**: Separates model from agent logic

2. **`tools=tools`**:
   - Available capabilities for the agent
   - Agent can call these to gather information
   - **Why**: Extensible design (easy to add tools)

3. **`system_prompt`**:
   - Defines agent's role and behavior
   - Instructions on when/how to use tools
   - **Why**: Guides agent decision-making

4. **`debug=True`**:
   - Prints agent's reasoning steps
   - Shows which tools are called
   - **Why**: Helps troubleshoot and understand agent behavior

**Why LangGraph pattern**:
- Modern agent architecture (replaces old ReAct agents)
- Better error handling and retry logic
- Supports complex workflows and state management
- More reliable tool calling

#### `run_agent(query)` Function

```python
def run_agent(query):
    result = agent_executor.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result
```

**Parameters**:
- `query` (str): User's request

**Returns**:
- Agent's response with full execution trace

**Why this structure**:

1. **Message format**:
   - Standard chat message structure
   - Supports multi-turn conversations
   - **Why**: Enables conversation history in future

2. **`agent_executor.invoke()`**:
   - Executes agent reasoning loop
   - Decides which tools to call
   - Synthesizes final answer
   - **Why invoke**: Synchronous execution for APIs

**Agent workflow**:
1. Receives user query
2. Reasons about which tools to use
3. Calls tools (search, retrieve, read)
4. Synthesizes information
5. Generates final response

**Difference from `answer_question()`**:
- `answer_question()`: Simple RAG (retrieve → answer)
- `run_agent()`: Complex reasoning (can search web, read pages, use multiple tools)

---

### 6. **app.py** - FastAPI Web Server

**Purpose**: Exposes the AI agent as a REST API endpoint.

**Why we use it**:
- Makes the agent accessible via HTTP
- Enables web/mobile app integration
- Supports concurrent requests
- Production-ready deployment

#### FastAPI Application

```python
app = FastAPI()
```
- **What it does**: Creates a web application instance
- **Why FastAPI**:
  - Async support (handles many requests)
  - Automatic API documentation (Swagger)
  - Fast performance (on par with Node.js)
  - Type validation built-in

#### `/ask` Endpoint

```python
@app.get("/ask")
def ask(q: str):
    answer = run_agent(q)
    return {"response": answer}
```

**What it does**: HTTP GET endpoint that accepts a question and returns an answer

**Parameters**:
- `q` (str): Query parameter from URL (e.g., `/ask?q=your+question`)

**Returns**:
- JSON object: `{"response": "agent's answer"}`

**Why this design**:

1. **GET method**:
   - Simple to test in browser
   - Can be bookmarked/cached
   - **Why not POST**: GET is simpler for Q&A use case

2. **Query parameter `q`**:
   - Easy to construct URLs
   - FastAPI auto-validates and extracts
   - **Why `q`**: Short, common convention

3. **`run_agent(q)`**:
   - Leverages full agent capabilities
   - Can use search, database, web reading
   - **Why not answer_question()**: More powerful, flexible

4. **JSON response**:
   - Standard API format
   - Easy to parse in any language
   - **Why**: Enables frontend integration

**Example usage**:
```
GET http://localhost:8080/ask?q=what are the latest AI frameworks?
```

**Auto-generated docs**:
- Visit `http://localhost:8080/docs` for interactive API documentation
- **Why**: FastAPI automatically creates OpenAPI/Swagger UI

---

## 🔍 Function Reference Summary

### Core Functions by Purpose

#### **Data Storage & Retrieval**
| Function | File | Purpose |
|----------|------|---------|
| `vector_store.add_texts()` | vectore_store.py | Store documents in vector database |
| `vector_store.similarity_search()` | vectore_store.py | Find similar documents by semantic meaning |
| `retrieve_context()` | agent.py | Get relevant context for a query |
| `retrieve_docs()` | tools.py | Tool wrapper for vector search |

#### **Web Data Collection**
| Function | File | Purpose |
|----------|------|---------|
| `search_tool.invoke()` | tools.py | Search web using Tavily API |
| `read_webpage()` | scraper.py | Extract text from URL |
| `read_page()` | tools.py | Tool wrapper for web scraping |
| `collect_research()` | rag.py | Complete pipeline: search → scrape → store |

#### **AI Reasoning & Responses**
| Function | File | Purpose |
|----------|------|---------|
| `answer_question()` | agent.py | Simple RAG-based Q&A |
| `run_agent()` | agent.py | Advanced agent with tool use |
| `llm.invoke()` | agent.py | Call Gemini AI directly |
| `ask()` | app.py | API endpoint for questions |

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.13+
- API Keys:
  - Google Gemini API key
  - Tavily API key

### Installation

1. **Clone the repository**:
```bash
cd "AI Research Assistant (Multi-Tool RAG Agent)"
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install langchain-community langchain-tavily python-dotenv
pip install langchain-huggingface langchain-google-genai
pip install langchain-chroma sentence-transformers
pip install requests beautifulsoup4 fastapi uvicorn
```

4. **Create `.env` file**:
```bash
touch .env
```

Add your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

5. **Get API Keys**:
- **Gemini**: https://makersuite.google.com/app/apikey
- **Tavily**: https://tavily.com (free tier available)

---

## 🚀 Usage Examples

### 1. Test Vector Database

```bash
python test_vector.py
```

**What it does**:
- Creates sample documents
- Stores them in ChromaDB
- Performs similarity search
- Prints matching results

**Expected output**:
```
AI agents are becoming very popular
AI agents are becoming very popular
LangChain helps build LLM applications
```

### 2. Test RAG Pipeline

```bash
python test_rag.py
```

**What it does**:
- Searches web for "latest AI agent frameworks"
- Scrapes top 5 results
- Stores content in vector database
- Reports number of documents stored

**Expected output**:
```
documents stored 4
```

### 3. Test Agent Q&A

```bash
python test_agent.py
```

**What it does**:
- Asks: "what are the latest AI agent frameworks?"
- Retrieves context from vector database
- Uses Gemini to generate answer
- Prints detailed response

**Expected output**: Comprehensive answer citing frameworks like LangChain, AutoGen, CrewAI, etc.

### 4. Run Web API

```bash
uvicorn app:app --reload --port 8080
```

**What it does**:
- Starts FastAPI server on port 8080
- Auto-reloads on code changes
- Serves `/ask` endpoint

**Test in browser**:
```
http://localhost:8080/ask?q=what%20is%20RAG?
```

**Interactive docs**:
```
http://localhost:8080/docs
```

---

## 📡 API Reference

### GET `/ask`

Ask a question to the AI research assistant.

**Parameters**:
- `q` (string, required): The question to ask

**Response**:
```json
{
  "response": "The agent's detailed answer with sources..."
}
```

**Example**:
```bash
curl "http://localhost:8080/ask?q=explain%20vector%20databases"
```

**Response Structure**:
The response includes:
- Direct answer to the question
- Citations from stored research
- Reasoning from the agent's tool usage

---

## 🧪 Testing Workflow

### Complete Testing Sequence

1. **Test basic vector operations**:
```bash
python test_vector.py
```
Verifies: embeddings, storage, similarity search

2. **Collect research data**:
```bash
python test_rag.py
```
Verifies: web search, scraping, storage pipeline

3. **Test Q&A**:
```bash
python test_agent.py
```
Verifies: retrieval, LLM integration, response generation

4. **Test API**:
```bash
uvicorn app:app --reload --port 8080
# In another terminal:
curl "http://localhost:8080/ask?q=test%20question"
```
Verifies: FastAPI, endpoint, full integration

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. **Module Import Errors**
```
ModuleNotFoundError: No module named 'langchain_community'
```
**Solution**: Install missing packages:
```bash
pip install langchain-community langchain-tavily
```

#### 2. **API Key Errors**
```
Did not find tavily_api_key, please add an environment variable
```
**Solution**: Check `.env` file exists and contains:
```
TAVILY_API_KEY=your_key_here
```

#### 3. **Embedding Dimension Mismatch**
```
Embedding dimension 384 does not match collection dimensionality 768
```
**Solution**: Delete old database and recreate:
```bash
rm -rf ./data
python test_rag.py
```

#### 4. **ChromaDB Telemetry Warnings**
```
Failed to send telemetry event ClientStartEvent
```
**Solution**: These are harmless warnings and can be ignored.

---

## 🎓 Learning Path

### Understanding the Flow

**For a simple question**:
1. User asks question via API
2. `answer_question()` retrieves relevant docs from vector DB
3. Gemini receives question + context
4. Generates answer based on stored knowledge

**For a complex research task**:
1. User asks question via API
2. Agent analyzes what information is needed
3. Agent calls `search_tool` to find web sources
4. Agent calls `read_page` to get detailed content
5. Agent calls `retrieve_docs` to check existing knowledge
6. Agent synthesizes all information
7. Returns comprehensive answer with sources

### Key Concepts

**RAG (Retrieval-Augmented Generation)**:
- Retrieve relevant documents → Augment prompt with context → Generate answer
- **Why**: More accurate than pure LLM (no hallucination)

**Vector Embeddings**:
- Text → Numbers that capture meaning
- Similar text = similar vectors
- **Why**: Enables semantic search (meaning-based, not keyword-based)

**Agent Pattern**:
- LLM decides which tools to use
- Executes tools based on user needs
- Reasons over results
- **Why**: Handles complex, multi-step tasks

---

## 📚 Further Enhancements

### Potential Improvements

1. **Add conversation memory**:
   - Store chat history
   - Enable follow-up questions
   - Maintain context across turns

2. **Expand tool set**:
   - Add calculator for math
   - Add code execution for analysis
   - Add database queries

3. **Improve error handling**:
   - Retry failed web requests
   - Validate inputs
   - Better error messages

4. **Optimize performance**:
   - Cache frequent queries
   - Batch embeddings
   - Use async for web requests

5. **Add authentication**:
   - API keys for users
   - Rate limiting
   - Usage tracking

---

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

To extend this project:
1. Add new tools in `tools.py`
2. Update agent prompt in `agent.py`
3. Test with relevant queries
4. Document your changes

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review function documentation
3. Test components individually
4. Check API logs for errors

---

**Built with**: LangChain, Google Gemini, ChromaDB, FastAPI, and HuggingFace 🚀
