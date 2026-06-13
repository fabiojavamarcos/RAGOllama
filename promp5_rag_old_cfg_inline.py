from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


# Initialize the embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    request_timeout=300.0,  # Increased timeout for large documents
)

# Initialize the LLM with optimized settings
llm = Ollama(
    model="llama3.2",  # Confirm with `ollama list`
    request_timeout=300.0,
    temperature=0.1,          # Lower temperature for more factual responses
)

# Set global configurations
Settings.embed_model = embed_model
Settings.llm = llm

def load_and_index_documents(data_dir):
    """Load documents and create vector index"""

    # Check if data directory exists
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please create it and add your PDF files.")
    print("data_dir", data_dir)

    # Load documents from the data folder
    docs = SimpleDirectoryReader(data_dir).load_data()

    if not docs:
        raise ValueError(f"No documents found in {data_dir}")
    print ("docs",docs)

    # Build vector index from documents
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

    return index, docs

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
    data_dir="data"
    try:
        # Load documents and create index
        index, docs = load_and_index_documents(data_dir)

        # Create query engine
        query_engine = create_query_engine(index)

        # Sample test queries
        test_queries = [
        "Summarize this document in a paragraph. Folder" + data_dir,
            '''You will perform an EVIDENCE-BASED ACADEMIC REVIEW of a scientific paper. IMPORTANT:
This evaluation REQUIRES TWO DOCUMENTS PROVIDED TOGETHER:
(1) The SUBMITTED PAPER : msr2026-paper78.pdf
(2) OTHER REVIEWS as an example : Review467.pdf

The Call for Papers is the primary normative reference and must guide all judgments.

GENERAL RULES:

- For each evaluative statement, explicitly indicate whether it is:
(a) directly supported by the paper, or
(b) an analytical inference you are making
- Do not invent external references or unsupported facts
- Explicitly state when information is insufficient for evaluation

Call for Papers (CfP):
Submissions will be evaluated according to the following criteria:

1. STRENGTHS
    - Clearly observable strengths grounded in the paper
    - Cite sections or elements supporting each strength
2. WEAKNESSES
    - Methodological, conceptual, or empirical limitations
    - Explain how each limitation affects validity, rigor, or impact
3. RELEVANCE: The extent to which the paper successfully argues or illustrates that its contributions help bridge a significant knowledge gap or tackle a crucial practical issue within the field of software engineering.
    
4. NOVELTY: How original the paper’s contributions are in comparison to existing knowledge or how significantly they contribute to the current body of knowledge. Note that this doesn’t discourage well-motivated replication studies.
    
5. SOUNDNESS: This aspect pertains to how well the paper’s contributions — whether they involve new methodologies, applications of existing techniques to unfamiliar problems, empirical studies, or other research methods — address the research questions posed and are backed by a thorough application of relevant research procedures. For short papers, the expectation is for more limited evaluations given their narrower scope.
    
6. PRESENTATION: How well-structured and clear the paper’s argumentation is, how clearly the contributions are articulated, the legibility of figures and tables, and the adequacy of English language usage. All papers should comply with the formatting instructions provided.
   
7. REPLICABILITY: The extent to which the paper’s claims can be independently verified through available replication packages and/or sufficient information included in the paper to understand how data was obtained, analyzed, and interpreted, or how a proposed technique works. All submissions are expected to adhere to the Open Science policy below.


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
