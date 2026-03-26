"""
Benchmark Script
----------------
Compares RAG performance with and without Contextual Retrieval.

Based on DataPizza research (Jan 2026):
- Contextual Retrieval improves recall, especially at low k values
- Best gains at k=5, works well with reranker
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
from src.generation.llm_client import OllamaClient
from src.rag.pipeline import RAGPipeline
from evaluation.metrics import evaluate_retrieval, evaluate_response


def load_test_questions(filepath: str = "evaluation/test_questions.json") -> list:
    """
    Load test questions from JSON file with ground truth.

    Returns:
        List of question dictionaries with expected_answer, expected_keywords, etc.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        questions = data.get("questions", [])

        # Convert format: map expected_answer_contains to expected_keywords for compatibility
        for q in questions:
            # Use expected_answer_contains as keywords, or expected_topics as fallback
            if "expected_answer_contains" in q:
                q["expected_keywords"] = q["expected_answer_contains"]
            elif "expected_topics" in q:
                q["expected_keywords"] = q["expected_topics"]
            else:
                q["expected_keywords"] = []

        print(f"Loaded {len(questions)} test questions from {filepath}")
        return questions

    except FileNotFoundError:
        print(f"Warning: {filepath} not found, using fallback questions")
        return _get_fallback_questions()
    except json.JSONDecodeError as e:
        print(f"Warning: Error parsing {filepath}: {e}, using fallback questions")
        return _get_fallback_questions()


def _get_fallback_questions() -> list:
    """Fallback questions if JSON file is not available."""
    return [
        {
            "question": "What are Infineon's main business segments?",
            "expected_keywords": ["automotive", "power", "sensor", "security"],
            "relevant_sources": ["infineon_company_overview.txt", "infineon_products_automotive.txt"]
        },
        {
            "question": "What products does Infineon make for electric vehicles?",
            "expected_keywords": ["power", "semiconductor", "IGBT", "MOSFET", "EV", "electric"],
            "relevant_sources": ["infineon_products_automotive.txt"]
        },
        {
            "question": "What is Infineon's sustainability strategy?",
            "expected_keywords": ["carbon", "emission", "environment", "green", "climate", "sustainable"],
            "relevant_sources": ["infineon_sustainability.txt"]
        },
        {
            "question": "What skills are needed for an internship at Infineon?",
            "expected_keywords": ["python", "machine learning", "data", "engineering", "programming"],
            "relevant_sources": ["infineon_careers_intern.txt"]
        },
        {
            "question": "What is Infineon's AI strategy?",
            "expected_keywords": ["AI", "artificial intelligence", "machine learning", "automation"],
            "relevant_sources": ["infineon_ai_strategy.txt"]
        },
    ]


# Load test questions from JSON file (with ground truth)
TEST_QUESTIONS = load_test_questions()


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


def evaluate_ground_truth(answer: str, expected_answer: str) -> dict:
    """
    Evaluate answer against ground truth using RECALL-based metrics.

    This measures: "Does the answer contain the key facts from the expected answer?"
    It does NOT penalize answers that are more complete/verbose than expected.

    Args:
        answer: Generated answer from RAG
        expected_answer: Ground truth answer from test_questions.json

    Returns:
        Dictionary with accuracy metrics
    """
    import re

    if not expected_answer or not answer:
        return {"accuracy": 0.0, "fact_recall": 0.0, "number_recall": 0.0, "entity_recall": 0.0}

    answer_lower = answer.lower()
    expected_lower = expected_answer.lower()

    # Remove common stop words for better comparison
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'and', 'or', 'but', 'if', 'then', 'than', 'as', 'that', 'this',
                  'it', 'its', 'their', 'they', 'we', 'our', 'your', 'which'}

    # === FACT RECALL ===
    # How many important words from expected_answer appear in the answer?
    # This is RECALL, not Jaccard - we don't penalize extra words in the answer
    expected_words = set(expected_lower.split()) - stop_words
    answer_words = set(answer_lower.split()) - stop_words

    if not expected_words:
        fact_recall = 1.0
    else:
        # Count how many expected words are found in answer
        found_words = expected_words & answer_words
        fact_recall = len(found_words) / len(expected_words)

    # === NUMBER RECALL ===
    # All numbers from expected answer should appear in the answer
    expected_numbers = set(re.findall(r'\d+(?:\.\d+)?', expected_answer))
    answer_numbers = set(re.findall(r'\d+(?:\.\d+)?', answer))

    if not expected_numbers:
        number_recall = 1.0
    else:
        found_numbers = expected_numbers & answer_numbers
        number_recall = len(found_numbers) / len(expected_numbers)

    # === ENTITY RECALL ===
    # Important entities (proper nouns) from expected should appear in answer
    # Extract capitalized words/phrases
    expected_entities = set(re.findall(r'[A-Z][a-zA-Z]+', expected_answer))
    answer_entities = set(re.findall(r'[A-Z][a-zA-Z]+', answer))

    # Also check for acronyms (all caps, 2+ letters)
    expected_acronyms = set(re.findall(r'\b[A-Z]{2,}\b', expected_answer))
    answer_acronyms = set(re.findall(r'\b[A-Z]{2,}\b', answer))

    expected_entities = expected_entities | expected_acronyms
    answer_entities = answer_entities | answer_acronyms

    if not expected_entities:
        entity_recall = 1.0
    else:
        found_entities = expected_entities & answer_entities
        entity_recall = len(found_entities) / len(expected_entities)

    # === COMBINED ACCURACY ===
    # Weighted combination: entities and numbers are more important than general words
    accuracy = (fact_recall * 0.3) + (number_recall * 0.35) + (entity_recall * 0.35)

    return {
        "accuracy": round(accuracy, 3),
        "fact_recall": round(fact_recall, 3),
        "number_recall": round(number_recall, 3),
        "entity_recall": round(entity_recall, 3)
    }


def run_benchmark(pipeline, test_name: str) -> dict:
    """Run benchmark on a pipeline."""
    print(f"\n{'='*50}")
    print(f"  Running: {test_name}")
    print(f"{'='*50}")

    results = []
    total_latency = 0
    total_keyword_score = 0
    total_faithfulness = 0
    total_precision = 0
    total_accuracy = 0
    has_ground_truth = False

    for i, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        expected_keywords = test.get("expected_keywords", [])
        relevant_sources = test.get("relevant_sources", [])
        expected_answer = test.get("expected_answer", "")

        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question[:50]}...")

        # Query
        start = time.time()
        result = pipeline.query(question)
        latency = time.time() - start

        # Evaluate keywords (original method)
        keyword_eval = evaluate_answer(result["answer"], expected_keywords)

        # Evaluate ground truth (NEW - if expected_answer is available)
        if expected_answer:
            has_ground_truth = True
            ground_truth_eval = evaluate_ground_truth(result["answer"], expected_answer)
        else:
            ground_truth_eval = {"accuracy": 0.0, "word_overlap": 0.0, "number_match": 0.0, "entity_match": 0.0}

        # Evaluate retrieval quality (from evaluation/metrics.py)
        retrieved_sources = [{"source": s} for s in result.get("sources", [])]
        retrieval_eval = evaluate_retrieval(retrieved_sources, relevant_sources)

        # Evaluate response quality (from evaluation/metrics.py)
        response_eval = evaluate_response(
            answer=result["answer"],
            context=result.get("context", "")
        )

        results.append({
            "question": question,
            "expected_answer": expected_answer[:100] + "..." if len(expected_answer) > 100 else expected_answer,
            "answer": result["answer"][:200] + "..." if len(result["answer"]) > 200 else result["answer"],
            "latency": latency,
            "keyword_score": keyword_eval["score"],
            "found_keywords": keyword_eval["found"],
            "missing_keywords": keyword_eval["missing"],
            "ground_truth": ground_truth_eval,
            "retrieval": {
                "precision": retrieval_eval["precision"],
                "recall": retrieval_eval["recall"],
                "f1": retrieval_eval["f1"]
            },
            "response": {
                "faithfulness": response_eval["faithfulness"],
                "length_score": response_eval["length_score"],
                "overall": response_eval["overall"]
            },
            "num_docs": result["num_docs"]
        })

        total_latency += latency
        total_keyword_score += keyword_eval["score"]
        total_faithfulness += response_eval["faithfulness"]
        total_precision += retrieval_eval["precision"]
        total_accuracy += ground_truth_eval["accuracy"]

        # Print with accuracy if ground truth available
        if expected_answer:
            print(f"  Accuracy: {ground_truth_eval['accuracy']:.0%} | Keywords: {keyword_eval['score']:.0%} | Faithfulness: {response_eval['faithfulness']:.0%} | Precision: {retrieval_eval['precision']:.0%} | Latency: {latency:.2f}s")
        else:
            print(f"  Keywords: {keyword_eval['score']:.0%} | Faithfulness: {response_eval['faithfulness']:.0%} | Precision: {retrieval_eval['precision']:.0%} | Latency: {latency:.2f}s")

    num_questions = len(TEST_QUESTIONS)

    benchmark_result = {
        "name": test_name,
        "avg_keyword_score": total_keyword_score / num_questions,
        "avg_faithfulness": total_faithfulness / num_questions,
        "avg_precision": total_precision / num_questions,
        "avg_latency": total_latency / num_questions,
        "total_latency": total_latency,
        "results": results
    }

    # Add accuracy metric only if ground truth was available
    if has_ground_truth:
        benchmark_result["avg_accuracy"] = total_accuracy / num_questions

    return benchmark_result


def print_comparison(baseline: dict, contextual: dict):
    """Print comparison between baseline and contextual retrieval."""
    print("\n")
    print("=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Baseline':<15} {'Contextual':<15} {'Diff':<15}")
    print("-" * 70)

    # Ground Truth Accuracy (NEW - if available)
    if "avg_accuracy" in baseline or "avg_accuracy" in contextual:
        b_acc = baseline.get("avg_accuracy", 0)
        c_acc = contextual.get("avg_accuracy", 0)
        acc_diff = c_acc - b_acc
        print(f"{'Ground Truth Accuracy':<25} {b_acc:.1%}{'':<8} {c_acc:.1%}{'':<8} {acc_diff:+.1%}")

    # Keyword Score
    b_kw = baseline.get("avg_keyword_score", baseline.get("avg_score", 0))
    c_kw = contextual.get("avg_keyword_score", contextual.get("avg_score", 0))
    kw_diff = c_kw - b_kw
    print(f"{'Keyword Score':<25} {b_kw:.1%}{'':<8} {c_kw:.1%}{'':<8} {kw_diff:+.1%}")

    # Faithfulness
    b_faith = baseline.get("avg_faithfulness", 0)
    c_faith = contextual.get("avg_faithfulness", 0)
    faith_diff = c_faith - b_faith
    print(f"{'Faithfulness':<25} {b_faith:.1%}{'':<8} {c_faith:.1%}{'':<8} {faith_diff:+.1%}")

    # Retrieval Precision
    b_prec = baseline.get("avg_precision", 0)
    c_prec = contextual.get("avg_precision", 0)
    prec_diff = c_prec - b_prec
    print(f"{'Retrieval Precision':<25} {b_prec:.1%}{'':<8} {c_prec:.1%}{'':<8} {prec_diff:+.1%}")

    # Average Latency
    latency_diff = contextual["avg_latency"] - baseline["avg_latency"]
    print(f"{'Avg Latency (s)':<25} {baseline['avg_latency']:.2f}{'':<11} {contextual['avg_latency']:.2f}{'':<11} {latency_diff:+.2f}")

    # Total Time
    print(f"{'Total Time (s)':<25} {baseline['total_latency']:.2f}{'':<11} {contextual['total_latency']:.2f}{'':<11}")

    print("\n" + "-" * 70)
    print("Per-Question Comparison:")
    print("-" * 70)

    for i, (b, c) in enumerate(zip(baseline["results"], contextual["results"]), 1):
        q = b["question"][:40] + "..." if len(b["question"]) > 40 else b["question"]

        # Get keyword scores (handle both old and new format)
        b_score = b.get("keyword_score", b.get("score", 0))
        c_score = c.get("keyword_score", c.get("score", 0))
        diff = c_score - b_score

        # Get ground truth accuracy (NEW)
        b_acc = b.get("ground_truth", {}).get("accuracy", 0)
        c_acc = c.get("ground_truth", {}).get("accuracy", 0)

        # Get faithfulness scores
        b_faith = b.get("response", {}).get("faithfulness", 0)
        c_faith = c.get("response", {}).get("faithfulness", 0)

        indicator = "✓" if diff > 0 else ("=" if diff == 0 else "✗")
        print(f"{i}. {q}")
        if b_acc > 0 or c_acc > 0:
            print(f"   Accuracy:     Baseline: {b_acc:.0%} | Contextual: {c_acc:.0%}")
        print(f"   Keywords:     Baseline: {b_score:.0%} | Contextual: {c_score:.0%} | {indicator} {diff:+.0%}")
        print(f"   Faithfulness: Baseline: {b_faith:.0%} | Contextual: {c_faith:.0%}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Overall improvement (average of all metrics)
    overall_diff = (kw_diff + faith_diff + prec_diff) / 3

    if overall_diff > 0:
        print(f"\n✓ Contextual Retrieval improved overall quality:")
        print(f"    Keywords:    {kw_diff:+.1%}")
        print(f"    Faithfulness: {faith_diff:+.1%}")
        print(f"    Precision:   {prec_diff:+.1%}")
    elif overall_diff < 0:
        print(f"\n✗ Contextual Retrieval decreased overall quality by {abs(overall_diff):.1%}")
    else:
        print(f"\n= No significant difference in quality")

    print(f"\n  Latency overhead: {latency_diff:+.2f}s per query")

    # Recommendation
    print("\n" + "-" * 70)
    if overall_diff > 0.05:  # >5% improvement
        print("Recommendation: USE Contextual Retrieval (quality gain > latency cost)")
    elif overall_diff < -0.05:  # >5% degradation
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

    # Create pipeline (reranker is created internally)
    baseline_pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=baseline_store,
        llm_client=llm_client,
        top_k=5
    )
    baseline_results = run_benchmark(baseline_pipeline, "Baseline (No Context)")

    # Save baseline results
    with open("./data/baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    print("\nBaseline results saved to: ./data/baseline_results.json")

    # Cleanup - unload reranker first (biggest memory user)
    baseline_pipeline.unload()
    shutil.rmtree(baseline_db, ignore_errors=True)
    del baseline_pipeline, baseline_store, embedder, llm_client
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

    # Create pipeline (reranker is created internally)
    contextual_pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=contextual_store,
        llm_client=llm_client,
        top_k=5
    )
    contextual_results = run_benchmark(contextual_pipeline, "Contextual Retrieval")

    # Save contextual results
    with open("./data/contextual_results.json", "w") as f:
        json.dump(contextual_results, f, indent=2)

    print("\nContextual results saved to: ./data/contextual_results.json")

    # Cleanup - unload reranker first (biggest memory user)
    contextual_pipeline.unload()
    shutil.rmtree(contextual_db, ignore_errors=True)
    del contextual_pipeline, contextual_store, embedder, llm_client
    gc.collect()

    print("Contextual benchmark complete!")
    return contextual_results


def run_quick_benchmark():
    """Run a quick benchmark with only 3 questions to compare baseline vs contextual."""
    global TEST_QUESTIONS

    print("\n" + "=" * 60)
    print("  QUICK BENCHMARK (3 Questions)")
    print("=" * 60)

    # Save original questions and use only first 3
    original_questions = TEST_QUESTIONS
    TEST_QUESTIONS = TEST_QUESTIONS[:3]

    print(f"\nUsing {len(TEST_QUESTIONS)} questions for quick comparison:")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"  {i}. {q['question'][:60]}...")

    try:
        # Initialize LLM client
        llm_client = OllamaClient(model="gemma3:4b")
        if not llm_client.is_available():
            print("\n❌ Ollama is not running! Start it with: ollama serve")
            return

        embedder = Embedder(model_name="qwen3-embedding:0.6b")

        # ==================== BASELINE ====================
        print("\n" + "-" * 60)
        print("  Phase 1: BASELINE (No Contextual Retrieval)")
        print("-" * 60)

        baseline_db = "./data/quick_baseline"
        print("\n[1/2] Ingesting documents (baseline)...")
        baseline_store = ingest_documents(llm_client, use_contextual=False, db_path=baseline_db)

        if not baseline_store:
            print("❌ Failed to ingest documents!")
            return

        baseline_pipeline = RAGPipeline(
            embedder=embedder,
            vector_store=baseline_store,
            llm_client=llm_client,
            top_k=5
        )

        baseline_results = run_benchmark(baseline_pipeline, "Baseline")

        # Cleanup baseline
        baseline_pipeline.unload()
        shutil.rmtree(baseline_db, ignore_errors=True)
        del baseline_pipeline, baseline_store
        gc.collect()
        time.sleep(1)

        # ==================== CONTEXTUAL ====================
        print("\n" + "-" * 60)
        print("  Phase 2: CONTEXTUAL RETRIEVAL")
        print("-" * 60)

        contextual_db = "./data/quick_contextual"
        print("\n[2/2] Ingesting documents (with context)...")
        contextual_store = ingest_documents(llm_client, use_contextual=True, db_path=contextual_db)

        if not contextual_store:
            print("❌ Failed to ingest documents!")
            return

        contextual_pipeline = RAGPipeline(
            embedder=embedder,
            vector_store=contextual_store,
            llm_client=llm_client,
            top_k=5
        )

        contextual_results = run_benchmark(contextual_pipeline, "Contextual")

        # Cleanup contextual
        contextual_pipeline.unload()
        shutil.rmtree(contextual_db, ignore_errors=True)
        del contextual_pipeline, contextual_store
        gc.collect()

        # ==================== COMPARISON ====================
        print_comparison(baseline_results, contextual_results)

        # Save quick results
        with open("./data/quick_benchmark_results.json", "w") as f:
            json.dump({
                "baseline": baseline_results,
                "contextual": contextual_results,
                "num_questions": len(TEST_QUESTIONS),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

        print("\n✓ Quick benchmark results saved to: ./data/quick_benchmark_results.json")

    finally:
        # Restore original questions
        TEST_QUESTIONS = original_questions
        gc.collect()


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
    parser.add_argument("--quick", action="store_true", help="Quick test with 3 questions (runs both)")

    args = parser.parse_args()

    if args.quick:
        run_quick_benchmark()
    elif args.baseline:
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
