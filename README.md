# DocQuery: RAG-Powered Document Q&A with Citations

<p align="center">
  <img src="docs/demo.gif" alt="Demo" width="800"/>
</p>

**Upload any PDF. Ask questions in plain English. Get cited answers.**

A Retrieval-Augmented Generation (RAG) system that lets you chat with your documents — research papers, financial reports, legal contracts, or textbooks — and get accurate, cited answers grounded in the actual text.

---

## Why This Project Exists

LLMs are powerful but they hallucinate. When you ask ChatGPT about a specific document, it might make up facts that sound right but aren't in the text. RAG solves this by forcing the model to retrieve relevant passages first, then generate answers based only on what it found. Every answer in DocQuery includes page-level citations so you can verify.

This isn't a wrapper around an API — it's a full RAG pipeline with chunking strategies, embedding models, vector search, reranking, and citation tracking built from scratch.

## Key Features

- **Multi-document support** — upload multiple PDFs and query across all of them
- **Citation tracking** — every answer shows which document and page it came from
- **Hybrid search** — combines semantic (vector) search with keyword (BM25) search for better retrieval
- **Configurable chunking** — supports fixed-size, sentence-based, and recursive chunking strategies
- **Conversation memory** — follow-up questions understand context from previous Q&A
- **Source highlighting** — see the exact passages used to generate each answer

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-document-qa.git
cd rag-document-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key (for the LLM — embeddings run locally)
export OPENAI_API_KEY="your-key-here"
# OR for fully local: no API key needed (uses Ollama)

# Launch the app
streamlit run streamlit_app/app.py
```

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              INGESTION PIPELINE              │
                    │                                             │
  PDF Upload ──────▶│  Extract ──▶ Chunk ──▶ Embed ──▶ Store    │
                    │  (PyMuPDF)  (Recursive) (HuggingFace) (Chroma) │
                    └─────────────────────────────────────────────┘
                                                    │
                                                    ▼
                    ┌─────────────────────────────────────────────┐
                    │               QUERY PIPELINE                │
                    │                                             │
  User Question ───▶│  Embed ──▶ Retrieve ──▶ Rerank ──▶ Generate │
                    │  Query    (Hybrid)    (Cross-Enc) (LLM+Cite)│
                    └─────────────────────────────────────────────┘
                                                    │
                                                    ▼
                                          Cited Answer + Sources
```

## How It Works

### 1. Document Ingestion
- PDFs are parsed with PyMuPDF, preserving page numbers and structure
- Text is split into overlapping chunks using recursive character splitting
- Each chunk is embedded using a local HuggingFace model (`all-MiniLM-L6-v2`)
- Embeddings are stored in ChromaDB with metadata (doc name, page number, chunk index)

### 2. Retrieval (Hybrid Search)
- User query is embedded using the same model
- **Semantic search**: finds chunks with similar meaning via cosine similarity in ChromaDB
- **Keyword search**: BM25 ranking catches exact matches that semantic search might miss
- Results from both are merged using Reciprocal Rank Fusion (RRF)

### 3. Reranking
- Top candidates are reranked using a cross-encoder model for more precise relevance scoring
- This step significantly improves answer quality over raw retrieval

### 4. Answer Generation
- Retrieved passages are formatted with source metadata
- LLM generates an answer constrained to the provided context
- Citation markers link each claim to specific source passages
- If the answer isn't in the documents, the model says so instead of hallucinating

## Project Structure

```
rag-document-qa/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── configs/
│   └── config.yaml           # All configurable parameters
├── data/
│   └── README.md              # Sample documents for testing
├── src/
│   ├── __init__.py
│   ├── document_loader.py     # PDF parsing and text extraction
│   ├── chunker.py             # Text chunking strategies
│   ├── embeddings.py          # Embedding model wrapper
│   ├── vector_store.py        # ChromaDB operations
│   ├── retriever.py           # Hybrid retrieval + reranking
│   ├── generator.py           # LLM answer generation with citations
│   └── rag_pipeline.py        # End-to-end orchestration
├── streamlit_app/
│   └── app.py                 # Interactive web interface
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_pipeline.py
└── docs/
    └── architecture.md
```

## Configuration

All parameters are in `configs/config.yaml`:

```yaml
chunking:
  strategy: "recursive"     # Options: fixed, sentence, recursive
  chunk_size: 512
  chunk_overlap: 50

retrieval:
  top_k: 10                 # Candidates from vector search
  rerank_top_k: 5           # Final passages after reranking
  use_hybrid: true           # Enable BM25 + semantic fusion

generation:
  model: "gpt-4o-mini"      # Or "ollama/llama3" for local
  temperature: 0.1
  max_tokens: 1000
```

## Running Fully Local (No API Key)

DocQuery supports fully local operation using Ollama:

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3

# Update config
# In configs/config.yaml, set: model: "ollama/llama3"

# Run — no API key needed
streamlit run streamlit_app/app.py
```

## What I'd Improve

- Add support for tables and images in PDFs (currently text-only)
- Implement query decomposition for complex multi-part questions
- Add evaluation benchmarks using RAGAS framework
- Build a feedback loop where users can rate answers to improve retrieval
- Add support for Word docs, HTML, and markdown in addition to PDFs
- Implement streaming responses for better UX on long answers

## License

MIT License — see [LICENSE](LICENSE) for details.
