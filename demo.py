"""
Demo Script
-----------
Interactive demo for the interview.

Usage:
    python demo.py
"""

import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_demo():
    """Main demo function."""
    
    print("\n" + "=" * 50)
    print("  INFINEON RAG DEMO")
    print("=" * 50)

    print("\nThis demo shows:")
    print("- Local LLM inference (data stays on-premise)")
    print("- Document ingestion and chunking")
    print("- Vector-based semantic search")
    print("- RAG question answering")
    
    input("\nPress Enter to start...")
    
    # Step 1: Check Ollama
    print("\n" + "-" * 50)
    print("STEP 1: Check Ollama")
    print("-" * 50)

    from src.generation.llm_client import OllamaClient
    
    client = OllamaClient(model="gemma3:4b")
    
    if not client.is_available():
        print("Ollama is not running!")
        print("Run: ollama serve")
        return
    
    print("Ollama is running")
    print(f"Models: {client.list_models()}")

    input("\nPress Enter...")

    # Step 2: Load Documents
    print("\n" + "-" * 50)
    print("STEP 2: Load Documents")
    print("-" * 50)

    from src.ingestion.document_loader import DocumentLoader
    from src.ingestion.chunker import TextChunker

    loader = DocumentLoader()
    documents = loader.load_directory("data/sample_docs")

    chunker = TextChunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk_documents(documents)

    print(f"\nSample chunk: \"{chunks[0]['text'][:100]}...\"")

    input("\nPress Enter...")

    # Step 3: Create Embeddings
    print("\n" + "-" * 50)
    print("STEP 3: Generate Embeddings")
    print("-" * 50)

    from src.embeddings.embedder import Embedder
    from src.vectorstore.chroma_store import ChromaStore

    embedder = Embedder(model_name="qwen3-embedding:0.6b")

    embeddings = []
    for i, chunk in enumerate(chunks):
        embeddings.append(embedder.embed(chunk["text"]))
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i+1}/{len(chunks)} chunks...")
    
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Store in temp database
    demo_db = "./data/demo_vectordb"
    if os.path.exists(demo_db):
        shutil.rmtree(demo_db)
    
    store = ChromaStore(persist_directory=demo_db, collection_name="demo")
    store.add_documents(chunks, embeddings)

    input("\nPress Enter...")

    # Step 4: RAG Q&A
    print("\n" + "-" * 50)
    print("STEP 4: RAG Question Answering")
    print("-" * 50)

    from src.retrieval.retriever import Retriever
    from src.retrieval.hf_reranker import HuggingFaceReranker
    from src.rag.pipeline import RAGPipeline
    
    reranker = HuggingFaceReranker()
    retriever = Retriever(embedder=embedder, vector_store=store, top_k=3, reranker=reranker)
    pipeline = RAGPipeline(retriever=retriever, llm_client=client, top_k=3)

    demo_questions = [
        "What are Infineon's main business segments?",
        "What products does Infineon make for electric vehicles?",
        "What is Infineon's AI strategy?",
    ]
    
    print("\nDemo questions:")
    for i, q in enumerate(demo_questions, 1):
        print(f"  {i}. {q}")
    
    print("\nType a number (1-3) or your own question")
    print("Type 'quit' to exit")
    
    while True:
        try:
            user_input = input("\nQuestion: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            # Get question
            if user_input.isdigit() and 1 <= int(user_input) <= len(demo_questions):
                question = demo_questions[int(user_input) - 1]
            else:
                question = user_input
            
            print(f"\nProcessing: \"{question}\"")

            result = pipeline.query(question)
            
            print("\n" + "-" * 40)
            print(f"Answer:\n{result['answer']}")
            print("-" * 40)
            print(f"Sources: {', '.join([Path(s).name for s in result['sources']])}")
            print(f"Latency: {result['latency_seconds']:.2f}s")
            
        except KeyboardInterrupt:
            break

    # Cleanup
    print("\n" + "=" * 50)
    print("DEMO COMPLETE")
    print("=" * 50)

    shutil.rmtree(demo_db, ignore_errors=True)
    print("\nThank you!")


if __name__ == "__main__":
    run_demo()
