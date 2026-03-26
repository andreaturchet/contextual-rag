"""
Quality Gates
-------------
Automatic quality checks for CI/CD pipelines.

Quality gates ensure the RAG system meets minimum standards
before a hypothetical deployment. If gates fail, deployment is blocked.
"""

from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class QualityGateError(Exception):
    """Raised when a quality gate fails."""
    pass


def check_quality_gates(
    evaluation_results: Dict,
    min_faithfulness: float = 0.7,
    max_latency_seconds: float = 150.0,
    min_retrieval_precision: float = 0.6
) -> Dict:
    """
    Check if evaluation results pass quality gates.
    
    Args:
        evaluation_results: Output from run_evaluation()
        min_faithfulness: Minimum faithfulness score
        max_latency_seconds: Maximum response time (default 150s for local LLMs on CPU)
        min_retrieval_precision: Minimum retrieval precision
        
    Returns:
        Dictionary with pass/fail status
    """
    logger.info("Checking quality gates...")
    
    gates = []
    all_passed = True
    
    # Gate 1: Faithfulness
    faithfulness = evaluation_results.get("average_faithfulness", 0)
    faith_passed = faithfulness >= min_faithfulness
    
    gates.append({
        "name": "Faithfulness",
        "passed": faith_passed,
        "value": faithfulness,
        "threshold": min_faithfulness,
        "message": f"Faithfulness: {faithfulness:.3f} (min: {min_faithfulness})"
    })
    
    if not faith_passed:
        all_passed = False
    
    # Gate 2: Latency
    latency = evaluation_results.get("average_latency_seconds", 0)
    latency_passed = latency <= max_latency_seconds
    
    gates.append({
        "name": "Latency",
        "passed": latency_passed,
        "value": latency,
        "threshold": max_latency_seconds,
        "message": f"Latency: {latency:.2f}s (max: {max_latency_seconds}s)"
    })
    
    if not latency_passed:
        all_passed = False
    
    # Gate 3: Retrieval Precision (if available)
    detailed = evaluation_results.get("detailed_results", [])
    precisions = [
        r["retrieval"]["precision"] 
        for r in detailed 
        if r["retrieval"]["precision"] is not None
    ]
    
    if precisions:
        avg_precision = sum(precisions) / len(precisions)
        precision_passed = avg_precision >= min_retrieval_precision
        
        gates.append({
            "name": "Retrieval Precision",
            "passed": precision_passed,
            "value": avg_precision,
            "threshold": min_retrieval_precision,
            "message": f"Precision: {avg_precision:.3f} (min: {min_retrieval_precision})"
        })
        
        if not precision_passed:
            all_passed = False
    
    return {
        "all_passed": all_passed,
        "gates": gates,
        "summary": {
            "total_gates": len(gates),
            "passed": sum(1 for g in gates if g["passed"]),
            "failed": sum(1 for g in gates if not g["passed"])
        }
    }


def print_quality_report(result: Dict) -> None:
    """Print a formatted quality report."""
    print("\n" + "=" * 60)
    print("QUALITY GATE REPORT")
    print("=" * 60)
    
    for gate in result["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"\n{status}: {gate['name']}")
        print(f"  {gate['message']}")
    
    print("\n" + "-" * 60)
    summary = result["summary"]
    print(f"TOTAL: {summary['passed']}/{summary['total_gates']} gates passed")
    
    if result["all_passed"]:
        print("\nALL QUALITY GATES PASSED")
    else:
        print("\nQUALITY GATES FAILED - Deployment blocked")
    
    print("=" * 60)


if __name__ == "__main__":
    import os
    from pathlib import Path

    print("Quality Gates Check")
    print("-" * 40)
    
    # Try to load real benchmark results
    base_path = Path(__file__).parent.parent / "data"
    contextual_path = base_path / "contextual_results.json"

    if contextual_path.exists():
        print(f"Loading results from: {contextual_path}")
        with open(contextual_path) as f:
            benchmark_data = json.load(f)

        # Convert benchmark format to evaluation format
        evaluation_results = {
            "average_faithfulness": benchmark_data.get("avg_faithfulness", 0),
            "average_latency_seconds": benchmark_data.get("avg_latency", 0),
            "detailed_results": [
                {"retrieval": r.get("retrieval", {"precision": None})}
                for r in benchmark_data.get("results", [])
            ]
        }

        print(f"\nBenchmark Data:")
        print(f"  Avg Faithfulness: {evaluation_results['average_faithfulness']:.1%}")
        print(f"  Avg Latency: {evaluation_results['average_latency_seconds']:.2f}s")

        result = check_quality_gates(evaluation_results)
        print_quality_report(result)
    else:
        print(f"No benchmark results found at {contextual_path}")
        print("Run 'python benchmark.py --contextual' first")


    print("\nTest completed")
