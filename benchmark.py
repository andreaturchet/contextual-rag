"""
Benchmark Script
----------------
Compares RAG performance with and without Contextual Retrieval.

Based on DataPizza research (Jan 2026):
- Contextual Retrieval improves recall, especially at low k values
- Best gains at k=5, works well with rerankers

Usage:
    python benchmark.py
"""

import os
import sys
import time
import shutil
import json
import gc
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.ingestion.contextual_chunker import ContextualChunker
from src.embeddings.embedder import Embedder
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.retriever import Retriever
from src.retrieval.hf_reranker import HuggingFaceReranker
from src.generation.llm_client import OllamaClient
from src.rag.pipeline import RAGPipeline


# Test questions with expected keywords in answers
TEST_QUESTIONS = [
    {
        "question": "What are Infineon's main business segments?",
        "expected_keywords": ["automotive", "power", "sensor", "security"]
    },
    {
        "question": "What products does Infineon make for electric vehicles?",
        "expected_keywords": ["power", "semiconductor", "chip", "IGBT", "MOSFET", "EV", "electric"]
    },
    {
        "question": "What is Infineon's sustainability strategy?",
        "expected_keywords": ["carbon", "emission", "environment", "green", "climate", "sustainable"]
    },
    {
        "question": "What skills are needed for an internship at Infineon?",
        "expected_keywords": ["python", "machine learning", "data", "engineering", "programming"]
    },
    {
        "question": "What is Infineon's AI strategy?",
        "expected_keywords": ["AI", "artificial intelligence", "machine learning", "automation"]
    },
]


def create_pipeline(embedder, store, llm_client, use_reranker=True):
    """Create a RAG pipeline."""
    reranker = HuggingFaceReranker() if use_reranker else None
    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
        top_k=5,
        reranker=reranker
    )
    return RAGPipeline(retriever=retriever, llm_client=llm_client, top_k=5)


def ingest_documents(llm_client, use_contextual: bool, db_path: str):
    """Ingest documents with or without contextual retrieval - MEMORY OPTIMIZED."""

    # Clean previous database
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Load documents
    loader = DocumentLoader()
    documents = loader.load_directory("data/sample_docs")

    if not documents:
        print("No documents found!")
        return None

    # Chunk documents
    chunker = TextChunker(chunk_size=500, overlap=50)

    # Create store and embedder
    store = ChromaStore(persist_directory=db_path, collection_name="benchmark")
    embedder = Embedder(model_name="qwen3-embedding:0.6b")

    # MEMORY OPTIMIZATION: Process ONE chunk at a time, embed, store, delete
    # This prevents accumulating all embeddings in memory

    if use_contextual:
        print("  Adding context to chunks (processing ONE chunk at a time)...")
        contextual = ContextualChunker(llm_client=llm_client, max_doc_length=2500, batch_size=2)

        total_chunks = 0
        for doc_idx, doc in enumerate(documents):
            doc_source = doc.get('source', 'unknown')
            print(f"  Document {doc_idx + 1}/{len(documents)}: {doc_source}")

            # Get document content and chunk it
            doc_content = doc.get("content", "")
            doc_chunks = chunker.chunk_documents([doc])

            # Add context to chunks (in-place)
            contextual.add_context_to_chunks(doc_chunks, doc_content)

            # Clear doc content immediately
            del doc_content
            gc.collect()

            # Process ONE CHUNK AT A TIME: embed and store immediately
            for chunk in doc_chunks:
                text = chunk.get("contextualized_text", chunk.get("text", ""))
                embedding = embedder.embed(text)
                store.add_documents([chunk], [embedding])

                # Clear immediately
                del embedding
                total_chunks += 1

            # Clear chunks after document is done
            del doc_chunks
            gc.collect()
            print(f"    Ingested {total_chunks} chunks so far...")

    else:
        # Non-contextual: still process one chunk at a time
        print("  Generating embeddings (one chunk at a time)...")
        chunks = chunker.chunk_documents(documents)
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            embedding = embedder.embed(text)
            store.add_documents([chunk], [embedding])

            # Clear embedding immediately
            del embedding

            if (i + 1) % 10 == 0:
                gc.collect()
                print(f"    Processed {i + 1}/{total} chunks...")

        del chunks
        gc.collect()

    return store


def evaluate_answer(answer: str, expected_keywords: list) -> dict:
    """Evaluate answer quality based on expected keywords."""
    answer_lower = answer.lower()

    found_keywords = []
    missing_keywords = []

    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0

    return {
        "score": score,
        "found": found_keywords,
        "missing": missing_keywords
    }


def run_benchmark(pipeline, test_name: str) -> dict:
    """Run benchmark on a pipeline."""
    print(f"\n{'='*50}")
    print(f"  Running: {test_name}")
    print(f"{'='*50}")

    results = []
    total_latency = 0
    total_score = 0

    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        expected = test["expected_keywords"]

        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question[:50]}...")

        # Query
        start = time.time()
        result = pipeline.query(question)
        latency = time.time() - start

        # Evaluate
        eval_result = evaluate_answer(result["answer"], expected)

        results.append({
            "question": question,
            "answer": result["answer"][:200] + "..." if len(result["answer"]) > 200 else result["answer"],
            "latency": latency,
            "score": eval_result["score"],
            "found_keywords": eval_result["found"],
            "missing_keywords": eval_result["missing"],
            "num_docs": result["num_docs"]
        })

        total_latency += latency
        total_score += eval_result["score"]

        print(f"  Score: {eval_result['score']:.0%} | Latency: {latency:.2f}s")

    avg_score = total_score / len(TEST_QUESTIONS)
    avg_latency = total_latency / len(TEST_QUESTIONS)

    return {
        "name": test_name,
        "avg_score": avg_score,
        "avg_latency": avg_latency,
        "total_latency": total_latency,
        "results": results
    }


def print_comparison(baseline: dict, contextual: dict):
    """Print comparison between baseline and contextual retrieval."""
    print("\n")
    print("=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Baseline':<15} {'Contextual':<15} {'Diff':<10}")
    print("-" * 60)

    # Average Score
    score_diff = contextual["avg_score"] - baseline["avg_score"]
    score_pct = (score_diff / baseline["avg_score"] * 100) if baseline["avg_score"] > 0 else 0
    print(f"{'Avg Answer Quality':<25} {baseline['avg_score']:.1%}{'':<8} {contextual['avg_score']:.1%}{'':<8} {score_diff:+.1%}")

    # Average Latency
    latency_diff = contextual["avg_latency"] - baseline["avg_latency"]
    print(f"{'Avg Latency (s)':<25} {baseline['avg_latency']:.2f}{'':<11} {contextual['avg_latency']:.2f}{'':<11} {latency_diff:+.2f}")

    # Total Time
    print(f"{'Total Time (s)':<25} {baseline['total_latency']:.2f}{'':<11} {contextual['total_latency']:.2f}{'':<11}")

    print("\n" + "-" * 60)
    print("Per-Question Comparison:")
    print("-" * 60)

    for i, (b, c) in enumerate(zip(baseline["results"], contextual["results"]), 1):
        q = b["question"][:40] + "..." if len(b["question"]) > 40 else b["question"]
        b_score = b["score"]
        c_score = c["score"]
        diff = c_score - b_score

        indicator = "✓" if diff > 0 else ("=" if diff == 0 else "✗")
        print(f"{i}. {q}")
        print(f"   Baseline: {b_score:.0%} | Contextual: {c_score:.0%} | {indicator} {diff:+.0%}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    if score_diff > 0:
        print(f"\n✓ Contextual Retrieval improved answer quality by {score_diff:.1%}")
    elif score_diff < 0:
        print(f"\n✗ Contextual Retrieval decreased answer quality by {abs(score_diff):.1%}")
    else:
        print(f"\n= No difference in answer quality")

    print(f"  Latency overhead: {latency_diff:+.2f}s per query")

    # Recommendation
    print("\n" + "-" * 60)
    if score_diff > 0.05:  # >5% improvement
        print("Recommendation: USE Contextual Retrieval (quality gain > latency cost)")
    elif score_diff < -0.05:  # >5% degradation
        print("Recommendation: DO NOT use Contextual Retrieval")
    else:
        print("Recommendation: Marginal difference - choose based on latency requirements")



def run_baseline_only():
    """Run only the baseline benchmark and save results."""
    print("\n" + "=" * 60)
    print("  BASELINE BENCHMARK (Contextual OFF)")
    print("=" * 60)

    llm_client = OllamaClient(model="gemma3:4b")
    if not llm_client.is_available():
        print("Ollama is not running!")
        return

    embedder = Embedder(model_name="qwen3-embedding:0.6b")

    baseline_db = "./data/benchmark_baseline"
    baseline_store = ingest_documents(llm_client, use_contextual=False, db_path=baseline_db)

    if not baseline_store:
        return

    # Create reranker separately so we can unload it
    reranker = HuggingFaceReranker()
    retriever = Retriever(
        embedder=embedder,
        vector_store=baseline_store,
        top_k=5,
        reranker=reranker
    )
    baseline_pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client, top_k=5)
    baseline_results = run_benchmark(baseline_pipeline, "Baseline (No Context)")

    # Save baseline results
    with open("./data/baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    print("\nBaseline results saved to: ./data/baseline_results.json")

    # Cleanup - unload reranker first (biggest memory user)
    reranker.unload()
    shutil.rmtree(baseline_db, ignore_errors=True)
    del baseline_pipeline, baseline_store, retriever, reranker, embedder, llm_client
    gc.collect()

    print("Baseline benchmark complete!")
    return baseline_results


def run_contextual_only():
    """Run only the contextual benchmark and save results."""
    print("\n" + "=" * 60)
    print("  CONTEXTUAL BENCHMARK (Contextual ON)")
    print("=" * 60)

    llm_client = OllamaClient(model="gemma3:4b")
    if not llm_client.is_available():
        print("Ollama is not running!")
        return

    embedder = Embedder(model_name="qwen3-embedding:0.6b")

    contextual_db = "./data/benchmark_contextual"
    contextual_store = ingest_documents(llm_client, use_contextual=True, db_path=contextual_db)

    if not contextual_store:
        return

    # Create reranker separately so we can unload it
    reranker = HuggingFaceReranker()
    retriever = Retriever(
        embedder=embedder,
        vector_store=contextual_store,
        top_k=5,
        reranker=reranker
    )
    contextual_pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client, top_k=5)
    contextual_results = run_benchmark(contextual_pipeline, "Contextual Retrieval")

    # Save contextual results
    with open("./data/contextual_results.json", "w") as f:
        json.dump(contextual_results, f, indent=2)

    print("\nContextual results saved to: ./data/contextual_results.json")

    # Cleanup - unload reranker first (biggest memory user)
    reranker.unload()
    shutil.rmtree(contextual_db, ignore_errors=True)
    del contextual_pipeline, contextual_store, retriever, reranker, embedder, llm_client
    gc.collect()

    print("Contextual benchmark complete!")
    return contextual_results


def compare_results():
    """Compare previously saved results."""
    baseline_file = "./data/baseline_results.json"
    contextual_file = "./data/contextual_results.json"

    if not os.path.exists(baseline_file):
        print(f"Missing {baseline_file}. Run: python benchmark.py --baseline")
        return
    if not os.path.exists(contextual_file):
        print(f"Missing {contextual_file}. Run: python benchmark.py --contextual")
        return

    with open(baseline_file) as f:
        baseline_results = json.load(f)
    with open(contextual_file) as f:
        contextual_results = json.load(f)

    print_comparison(baseline_results, contextual_results)

    # Save combined results
    with open("./data/benchmark_results.json", "w") as f:
        json.dump({
            "baseline": baseline_results,
            "contextual": contextual_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark: Contextual Retrieval Comparison")
    parser.add_argument("--baseline", action="store_true", help="Run only baseline benchmark (contextual OFF)")
    parser.add_argument("--contextual", action="store_true", help="Run only contextual benchmark (contextual ON)")
    parser.add_argument("--compare", action="store_true", help="Compare saved results")
    parser.add_argument("--all", action="store_true", help="Run both sequentially with memory cleanup")

    args = parser.parse_args()

    if args.baseline:
        run_baseline_only()
    elif args.contextual:
        run_contextual_only()
    elif args.compare:
        compare_results()
    elif args.all:
        print("\n" + "=" * 60)
        print("  RAG BENCHMARK: Running Both Sequentially")
        print("=" * 60)
        print("\nThis will run one benchmark at a time to save memory.\n")

        # Run baseline first
        run_baseline_only()

        # Force garbage collection
        gc.collect()
        time.sleep(2)

        # Run contextual
        run_contextual_only()

        # Compare
        compare_results()
    else:
        print("Usage:")
        print("  python benchmark.py --baseline    # Run baseline only (contextual OFF)")
        print("  python benchmark.py --contextual  # Run contextual only (contextual ON)")
        print("  python benchmark.py --compare     # Compare saved results")
        print("  python benchmark.py --all         # Run both sequentially")


if __name__ == "__main__":
    main()
