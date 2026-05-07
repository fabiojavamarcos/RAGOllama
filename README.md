# RAGOllama
Example using RAG with ollama

mkdir RAGOllama

cd RAGOllama

ollama --version

ollama pull nomic-embed-text

ollama pull llama3.1:latest

ollama serve

<this will lock the terminal. Don't open ollama desktop, or the port will be busy>

cd Documents

cd GitHub

cd RAGOllama

vim requirements.txt

<add the python libraries here>

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

ls

mkdir data

<add PDF documents to the data folder>

vim test_rag.py

<copy the code> 

python test_rag.py
