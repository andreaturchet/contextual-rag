"""
Evaluation Module
=================
This module contains tools for evaluating the RAG system.

Components:
- metrics: Functions to calculate retrieval and generation quality
- quality_gates: CI/CD checks to ensure quality standards
- test_questions.json: Test dataset for evaluation
"""

from .metrics import evaluate_retrieval, evaluate_response
from .quality_gates import check_quality_gates

__all__ = ["evaluate_retrieval", "evaluate_response", "check_quality_gates"]
