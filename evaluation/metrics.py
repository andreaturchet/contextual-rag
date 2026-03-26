"""
Evaluation Metrics
------------------
Functions for evaluating the RAG system.

Measures:
1. Retrieval Quality: Are the right documents found?
2. Generation Quality: Are the answers good?
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def evaluate_retrieval(
    retrieved_docs: List[Dict],
    relevant_doc_ids: List[str]
) -> Dict[str, float]:
    """
    Evaluate retrieval quality.
    
    Measures:
    - Precision: Fraction of retrieved docs that are relevant
    - Recall: Fraction of relevant docs that were retrieved
    - F1: Harmonic mean of precision and recall
    
    Args:
        retrieved_docs: Documents returned by retriever
        relevant_doc_ids: IDs that should have been retrieved
        
    Returns:
        Dictionary with precision, recall, f1
    """
    import os

    # Get IDs of retrieved documents (extract filename only from paths)
    retrieved_ids = set()
    for doc in retrieved_docs:
        doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("source", "")
        # Extract just the filename from full path
        filename = os.path.basename(doc_id) if doc_id else ""
        if filename:
            retrieved_ids.add(filename)

    # Also extract just filenames from relevant_doc_ids
    relevant_ids = set(os.path.basename(rid) for rid in relevant_doc_ids)

    # Handle empty cases
    if not retrieved_ids and not relevant_ids:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not retrieved_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not relevant_ids:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    
    # Calculate metrics
    true_positives = len(retrieved_ids & relevant_ids)
    
    precision = true_positives / len(retrieved_ids)
    recall = true_positives / len(relevant_ids)
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3)
    }


def evaluate_response(
    answer: str,
    context: str,
    expected_answer: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate response quality.
    
    Measures:
    - Faithfulness: Is answer based on context?
    - Length score: Is answer appropriately detailed?
    
    Args:
        answer: Generated answer
        context: Retrieved context
        expected_answer: Optional expected answer
        
    Returns:
        Dictionary with scores
    """
    # Empty answer check
    if not answer or len(answer.strip()) < 10:
        return {
            "faithfulness": 0.0,
            "length_score": 0.0,
            "overall": 0.0
        }
    
    # Faithfulness: word overlap between answer and context
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be",
                  "of", "to", "in", "for", "on", "with", "at", "by",
                  "and", "but", "if", "or", "this", "that"}
    
    context_words = set(context.lower().split()) - stop_words
    answer_words = set(answer.lower().split()) - stop_words
    
    if answer_words:
        overlap = len(answer_words & context_words) / len(answer_words)
        faithfulness = min(overlap * 1.5, 1.0)
    else:
        faithfulness = 0.0
    
    # Length score
    answer_length = len(answer)
    if answer_length < 50:
        length_score = 0.5
    elif answer_length < 500:
        length_score = 1.0
    else:
        length_score = 0.7
    
    # Check for uncertainty
    uncertainty = ["i don't know", "not enough information", "cannot answer"]
    is_uncertain = any(phrase in answer.lower() for phrase in uncertainty)
    
    if is_uncertain:
        overall = 0.3
    else:
        overall = (faithfulness + length_score) / 2
    
    return {
        "faithfulness": round(faithfulness, 3),
        "length_score": length_score,
        "is_uncertain": is_uncertain,
        "overall": round(overall, 3)
    }


def run_evaluation(
    pipeline,
    test_questions: List[Dict]
) -> Dict:
    """
    Run evaluation on test questions.
    
    Args:
        pipeline: RAGPipeline instance
        test_questions: List of test cases
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating {len(test_questions)} questions...")
    
    results = []
    
    for i, test in enumerate(test_questions):
        question = test.get("question", "")
        relevant_sources = test.get("relevant_sources", [])
        
        response = pipeline.query(question)
        
        if relevant_sources:
            retrieval_scores = evaluate_retrieval(
                response.get("sources", []),
                relevant_sources
            )
        else:
            retrieval_scores = {"precision": None, "recall": None, "f1": None}
        
        response_scores = evaluate_response(
            response.get("answer", ""),
            response.get("context", "")
        )
        
        results.append({
            "question": question,
            "retrieval": retrieval_scores,
            "response": response_scores,
            "latency": response.get("latency_seconds", 0)
        })
    
    # Calculate averages
    avg_faithfulness = sum(r["response"]["faithfulness"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    
    return {
        "num_questions": len(test_questions),
        "average_faithfulness": round(avg_faithfulness, 3),
        "average_latency_seconds": round(avg_latency, 3),
        "detailed_results": results
    }


if __name__ == "__main__":
    print("Testing Evaluation Metrics")
    print("-" * 40)
    
    # Test retrieval
    retrieved = [
        {"source": "doc1.txt"},
        {"source": "doc2.txt"},
        {"source": "doc3.txt"}
    ]
    relevant = ["doc1.txt", "doc2.txt"]
    
    scores = evaluate_retrieval(retrieved, relevant)
    print(f"Retrieval scores: {scores}")
    
    # Test response
    context = "Infineon is a German semiconductor company founded in 1999."
    answer = "Infineon is a German semiconductor company."
    
    scores = evaluate_response(answer, context)
    print(f"Response scores: {scores}")
    
    print("\nTest completed")
