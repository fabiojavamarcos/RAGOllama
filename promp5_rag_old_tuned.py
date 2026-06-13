from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


# Initialize the embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    request_timeout=300.0,  # Increased timeout for large documents
    context_window=4096
)

# Initialize the LLM with optimized settings
llm = Ollama(
    model="llama3.1:latest",  # Confirm with `ollama list`
    request_timeout=300.0,
    temperature=0.1,          # Lower temperature for more factual responses
)

# Set global configurations
Settings.embed_model = embed_model
Settings.llm = llm
# Optimize text parsing for a 16GB RAM constraint
Settings.chunk_size = 512   # Smaller text fragments per chunk
Settings.chunk_overlap = 50 # Smooth continuity between chunks

def load_and_index_documents(data_dir="/Users/fabiomar/Documents/GitHub/RAGOllama/data"):
    """Load documents and create vector index"""

    # Check if data directory exists
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please create it and add your PDF files.")

    # Load documents from the data folder
    docs = SimpleDirectoryReader(data_dir).load_data()

    if not docs:
        raise ValueError(f"No documents found in {data_dir}")


    # Build vector index from documents
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

    return index

def create_query_engine(index, similarity_top_k=3):
    """Create query engine with specified retrieval parameters"""

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k,  # Number of relevant chunks to retrieve
        response_mode="compact"             # Compact response generation
    )

    return query_engine

def test_rag_system():
    """Test the RAG system with sample queries"""

    try:
        # Load documents and create index
        index = load_and_index_documents()

        # Create query engine
        query_engine = create_query_engine(index)

        # Sample test queries
        test_queries = [
            '''You will perform an EVIDENCE-BASED ACADEMIC REVIEW of a scientific paper. IMPORTANT:
This evaluation REQUIRES TWO DOCUMENTS PROVIDED TOGETHER:
(1) The SUBMITTED PAPER 
(2) The CALL FOR PAPERS (CfP)

The Call for Papers is the primary normative reference and must guide all judgments.

GENERAL RULES:

- For each evaluative statement, explicitly indicate whether it is:
(a) directly supported by the paper, or
(b) an analytical inference you are making
- Do not invent external references or unsupported facts
- Explicitly state when information is insufficient for evaluation

Evaluate the paper according to the following categories, using their formal definitions:

1. STRENGTHS
    - Clearly observable strengths grounded in the paper
    - Cite sections or elements supporting each strength
2. WEAKNESSES
    - Methodological, conceptual, or empirical limitations
    - Explain how each limitation affects validity, rigor, or impact
3. RELEVANCE
    - Alignment and contribution to the field targeted by the CfP
    - Explicit assumptions required for relevance to materialize
4. NOVELTY
    - Originality relative to the described state of the art
    - Distinguish conceptual, methodological, and applied novelty
5. RIGOR
    - Soundness and appropriateness of the methodology
    - Consistency between research questions, methods, and conclusions
    - Judge rigor relative to the study type
6. PRESENTATION
    - Writing clarity, structure, figures, and tables
    - Assess whether presentation issues hinder comprehension
7. VERIFIABILITY & TRANSPARENCY
    - Sufficiency of detail for understanding, verification, or replication
    - Transparency of data, methods, and analytical decisions

Conclude with:

- A balanced synthesis of the paper’s fit to the CfP
- Key risks and key strengths of the work

Use cautious, formal academic language throughout.''',
        ]

        print("RAG System Test Results")
        print("=" * 50)

        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            print("-" * 40)

            try:
                response = query_engine.query(query)
                print(f"Response: {response}")
                print(f"Status: SUCCESS")
            except Exception as e:
                print(f"Error: {str(e)}")
                print(f"Status: FAILED")

            print("-" * 40)

        return True

    except Exception as e:
        print(f"System Error: {str(e)}")
        return False

# Main execution
if __name__ == "__main__":

    print("Starting RAG Pipeline Test...")

    # Test the complete system
    success = test_rag_system()

    if success:
        print("\nRAG system is working correctly!")
        print("You can now use the query_engine to ask questions about your documents.")
    else:
        print("\nRAG system test failed. Check the error messages above.")
