"""
RAG Module
==========
This module contains the main RAG (Retrieval-Augmented Generation) pipeline.

The RAG pipeline ties everything together:
1. Takes a user question
2. Retrieves relevant documents
3. Generates an answer using the LLM

This is the main entry point for the application.
"""

from .pipeline import RAGPipeline

__all__ = ["RAGPipeline"]
