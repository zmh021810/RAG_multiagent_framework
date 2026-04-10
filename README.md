# RAG_multiagent_framework
use RAG, vector database, chunk and overlap,  hybrid-search, reranking to build a RAG_multiagent_framework
You need to build a .env file under the same path of these python code, and in this .env file, write ur omwn key
PINECONE_API_KEY=......
GROQ_API_KEY=.......

Go to pinecone and groq website to create the vector database and LLM model for you. For the vector database, remember to
set the dimension to 384 and the Metric is dotproduct

ingest_data.py is used to input data from the data folder
clean_pinecone.py is used to clean the data
agent_graph.py is for the multi-agent framework
