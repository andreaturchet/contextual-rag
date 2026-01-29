"""
Retrieval Module
================
This module handles retrieving relevant documents for a query.

The retriever combines:
- Embedder (to convert query to vector)
- Vector Store (to find similar documents)

It's a convenience layer that makes searching easy.
"""

from .retriever import Retriever

__all__ = ["Retriever"]
