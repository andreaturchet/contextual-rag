# Contextual RAG Pipeline

A local Retrieval-Augmented Generation system that implements Anthropic's Contextual Retrieval technique for improved document question answering. All data processing and inference run entirely on-premise using Ollama, ensuring that no information leaves the local network.

## Project Description

This project builds a complete RAG pipeline designed for enterprise document retrieval and question answering. It addresses a core limitation of standard RAG systems: individual text chunks often lose important context when separated from their source document, which degrades retrieval accuracy.

The system implements Contextual Retrieval, a technique introduced by Anthropic in September 2024, where an LLM generates concise summaries of key facts for each chunk before embedding. These enriched representations lead to more accurate semantic search results. The pipeline also integrates a cross-encoder reranker (Qwen3-Reranker) that rescores candidate documents after initial retrieval, further improving the relevance of results passed to the generation model.

The codebase includes a full benchmarking framework that compares baseline RAG against Contextual RAG across multiple evaluation metrics, including ground truth accuracy, keyword recall, faithfulness, and retrieval precision. All benchmarks are designed to run on systems with limited resources (16GB RAM) through aggressive memory management and single-chunk processing.

### Key Capabilities

- **Contextual Retrieval**: LLM-generated context (key facts, entities, numbers) is prepended to each chunk before embedding, improving retrieval relevance
- **Cross-Encoder Reranking**: A Hugging Face reranker (Qwen3-Reranker-0.6B) rescores retrieved documents for higher precision
- **Local Inference**: All models run locally through Ollama -- no external API calls, no data exfiltration
- **Benchmarking Suite**: Side-by-side comparison of baseline vs. contextual retrieval with detailed per-question metrics
- **Memory-Optimized Processing**: Single-chunk embedding and storage pipeline designed for constrained environments
- **Multiple Interfaces**: Interactive CLI demo, programmatic main script, and REST API via FastAPI
- **Docker Deployment**: Production-ready containerized setup with docker-compose

## Architecture

```
Documents --> [Loader] --> [Chunker] --> [Contextual Chunker] --> [Embedder] --> [ChromaDB]
                                              |
                                         LLM (key facts
                                          extraction)

Query --> [Embedder] --> [Vector Search] --> [Reranker] --> [LLM] --> Answer
                              |
                          [ChromaDB]
```

**Models used:**
- LLM: gemma3:4b (generation and contextual chunking)
- Embeddings: Gemma3-based embedding model (768 dimensions)
- Reranker: Qwen3-Reranker-0.6B (cross-encoder reranking)

## Project Structure

```
contextual-rag/
|-- src/
|   |-- ingestion/          # Document loading, chunking, and contextual enrichment
|   |-- embeddings/         # Text embedding via Ollama
|   |-- vectorstore/        # ChromaDB vector storage
|   |-- retrieval/          # Semantic search and cross-encoder reranking
|   |-- generation/         # Ollama LLM client
|   +-- rag/                # Main RAG pipeline orchestration
|-- evaluation/             # Quality metrics, test questions, and quality gates
|-- deployment/             # Dockerfile and docker-compose configuration
|-- data/sample_docs/       # Sample enterprise documents for testing
|-- benchmark.py            # Full benchmark comparing baseline vs. contextual RAG
|-- benchmark_full.py       # Extended benchmark with additional test configurations
|-- demo.py                 # Step-by-step interactive demo
|-- main.py                 # CLI entry point for ingestion and querying
+-- config.yaml             # Centralized configuration for all components
```

## Quick Start

### Prerequisites

- Python 3.10+
- Ollama installed and running

### Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama and pull required models
ollama serve
ollama pull gemma3:4b
ollama pull qwen3-embedding:0.6b

# 3. Run the interactive demo
python demo.py

# Or start the main interactive session
python main.py
```

### Docker Deployment

```bash
# Start all services with one command
docker-compose -f deployment/docker-compose.yml up --build

# Wait for models to download (~5 min on first run)
# API docs available at: http://localhost:8000/docs
```

## Benchmarking

The benchmark suite runs the same set of test questions against both a baseline RAG system and the contextual retrieval variant, then produces a side-by-side comparison.

```bash
# Quick benchmark (3 questions, runs both pipelines)
python benchmark.py --quick

# Run baseline and contextual separately (lower memory usage)
python benchmark.py --baseline
python benchmark.py --contextual
python benchmark.py --compare

# Run everything sequentially
python benchmark.py --all
```

Evaluation metrics include:
- **Ground Truth Accuracy**: Weighted combination of fact recall, number recall, and entity recall against expected answers
- **Keyword Score**: Presence of expected domain-specific terms in the generated answer
- **Faithfulness**: Degree to which the answer is grounded in the retrieved context
- **Retrieval Precision**: Fraction of retrieved documents that match known relevant sources

## Configuration

All settings are centralized in `config.yaml`:

```yaml
llm:
  model_name: "gemma3:4b"
  temperature: 0.1

embeddings:
  model_name: "embeddinggemma"

chunking:
  chunk_size: 500
  chunk_overlap: 50
  contextual:
    enabled: true
    batch_size: 5

retrieval:
  top_k: 5

reranker:
  enabled: true
  model_name: "Qwen/Qwen3-Reranker-0.6B"
```

## References

- Anthropic, "Contextual Retrieval" (September 2024): https://www.anthropic.com/engineering/contextual-retrieval
- DataPizza research on contextual retrieval effectiveness (January 2026)

## Author

Andrea Turchet
