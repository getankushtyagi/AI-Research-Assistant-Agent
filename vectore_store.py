from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# create embedding model 
# model_name = "nomic-ai/nomic-embed-text-v1"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {
    'device': 'cpu',
    'trust_remote_code':True
}

encode_kwargs = {'normalize_embeddings': True}

embedding_model = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,    
)

# creating a vector db and store the data locally 
vector_store=Chroma(
    collection_name="research_agent",
    embedding_function=embedding_model,
    persist_directory='./data'
)

# to test this functionality works fine run the test_vector file