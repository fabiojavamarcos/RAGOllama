# RAGOllama

To run test_rag.py

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

-------

To run similarity.py

pip install pandas numpy scikit-learn rapidfuzz sentence-transformers spacy bert-score nltk rouge-score
python -m spacy download en_core_web_md

CSV file format

text_a,text_b
"fix database connection error","resolve issue with database connectivity"
"update login button color","improve neural network model accuracy"


