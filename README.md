# Secure RAG Pipeline

Local RAG (Retrieval-Augmented Generation) system for enterprise documents. All data stays on-premise.

## Features

- **Local LLM inference** with Ollama (no data leaves your network)
- **Document support**: PDF, TXT, Markdown
- **Vector search** with ChromaDB
- **REST API** with FastAPI
- **Docker deployment** ready

## Quick Start

### Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama and pull models
ollama serve
ollama pull gemma3:4b
ollama pull qwen3-embedding:0.6b

# 3. Run interactive demo
python demo.py

# Or start API server
python api.py
```

### Docker (Recommended)

```bash
# Start everything with one command
docker-compose -f deployment/docker-compose.yml up --build

# Wait for models to download (~5 min first time)
# Then open: http://localhost:8000/docs
```

## Usage

### Interactive Demo

```bash
python demo.py
```

### REST API

Start the server:
```bash
python api.py
```

Then use the API:

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Infineon?"}'

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"docs_path": "data/sample_docs"}'
```

Interactive API docs: http://localhost:8000/docs

### CLI

```bash
python main.py --help
python main.py ingest data/sample_docs
python main.py query "What products does Infineon make?"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Question → [Embedder] → [Vector Search] → [LLM] → Answer   │
│                              ↑                              │
│                         [ChromaDB]                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Models:
  • LLM: gemma3:4b (generation)
  • Embeddings: qwen3-embedding:0.6b (1024 dimensions)
```

## Project Structure

```
secure-llmops-pipeline/
├── src/
│   ├── ingestion/      # Document loading & chunking
│   ├── embeddings/     # Text embeddings (Ollama)
│   ├── vectorstore/    # ChromaDB storage
│   ├── retrieval/      # Semantic search
│   ├── generation/     # LLM client
│   └── rag/            # Main pipeline
├── evaluation/         # Quality metrics
├── deployment/         # Docker configuration
├── data/sample_docs/   # Sample documents
├── api.py              # REST API (FastAPI)
├── demo.py             # Interactive demo
├── main.py             # CLI interface
└── config.yaml         # Configuration
```

## Configuration

Edit `config.yaml`:

```yaml
llm:
  model_name: "gemma3:4b"
  temperature: 0.1

embeddings:
  model_name: "qwen3-embedding:0.6b"

chunking:
  chunk_size: 500
  chunk_overlap: 50

retrieval:
  top_k: 5
```

## Author

Andrea Turchet - Infineon MLOps Engineer Intern
