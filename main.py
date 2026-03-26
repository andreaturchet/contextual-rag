"""
Main Script
-----------
Run the RAG system interactively.

Usage:
    python main.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.ingestion.contextual_chunker import ContextualChunker
from src.embeddings.embedder import Embedder
from src.vectorstore.chroma_store import ChromaStore
from src.generation.llm_client import OllamaClient
from src.rag.pipeline import RAGPipeline

# Set to True to use Contextual Retrieval (adds LLM context to chunks)
USE_CONTEXTUAL_RETRIEVAL = True# Set True for better retrieval, False for faster ingestion


def ingest_documents(llm_client=None):
    """
    Ingest documents into vector store.

    If USE_CONTEXTUAL_RETRIEVAL is True, adds LLM-generated context
    to each chunk before embedding (Anthropic's technique).
    """
    print("\n[Ingesting documents...]")

    # Load documents
    loader = DocumentLoader()
    documents = loader.load_directory("data/sample_docs")

    if not documents:
        print("No documents found")
        return
    
    # Chunk documents
    chunker = TextChunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk_documents(documents)
    
    # Optional: Add context to chunks (Contextual Retrieval)
    if USE_CONTEXTUAL_RETRIEVAL and llm_client:
        print("\n[Adding context to chunks (Contextual Retrieval)...]")
        contextual = ContextualChunker(llm_client=llm_client)
        # Group chunks by source document
        for doc in documents:
            doc_chunks = [c for c in chunks if c.get("source") == doc.get("source")]
            if doc_chunks:
                contextualized = contextual.add_context_to_chunks(doc_chunks, doc.get("content", ""))
                # Update chunks with contextualized versions
                for i, chunk in enumerate(chunks):
                    if chunk.get("source") == doc.get("source"):
                        for ctx_chunk in contextualized:
                            if chunk.get("chunk_index") == ctx_chunk.get("chunk_index"):
                                chunks[chunks.index(chunk)] = ctx_chunk
                                break

    # Generate embeddings
    # For Contextual Retrieval: embed the contextualized_text
    embedder = Embedder(model_name="qwen3-embedding:0.6b")
    embeddings = []
    for i, chunk in enumerate(chunks):
        # Use contextualized_text if available, otherwise use text
        text_to_embed = chunk.get("contextualized_text", chunk.get("text", ""))
        embeddings.append(embedder.embed(text_to_embed))
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks...")
    
    # Store in vector database
    store = ChromaStore()
    store.add_documents(chunks, embeddings)
    
    print(f"Done: {len(chunks)} chunks indexed")


def main():
    """Main entry point."""
    print("\n" + "=" * 50)
    print("  RAG SYSTEM")
    print("=" * 50)

    print("\nInitializing...")

    # Check Ollama
    llm_client = OllamaClient(model="gemma3:4b")
    if not llm_client.is_available():
        print("\nOllama is not running!")
        print("1. Run: ollama serve")
        print("2. Run: ollama pull gemma3:4b")
        return
    print("  Ollama OK")

    # Initialize components
    embedder = Embedder(model_name="qwen3-embedding:0.6b")
    store = ChromaStore()
    print(f"  Vector store: {store.count()} documents")
    
    # Auto-ingest if empty
    if store.count() == 0:
        ingest_documents(llm_client)  # Pass LLM for contextual retrieval
        store = ChromaStore()

    # Initialize pipeline (retriever and reranker are created internally)
    pipeline = RAGPipeline(embedder=embedder, vector_store=store, llm_client=llm_client, top_k=5)
    print("  Pipeline ready")
    
    print("\n" + "=" * 50)
    print("Type a question (or 'quit' to exit)")
    print("=" * 50)

    # Interactive loop
    while True:
        try:
            question = input("\nQuestion: ").strip()

            if not question:
                continue
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            result = pipeline.query(question)

            print(f"\nAnswer:\n{result['answer']}")
            print(f"\n{'─' * 40}")
            print(f"Sources: {', '.join(result['sources'])}")
            print(f"Latency: {result['latency_seconds']:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break



if __name__ == "__main__":
    main()
