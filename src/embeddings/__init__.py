"""
Embeddings Module
=================
This module handles converting text into numerical vectors (embeddings).

What are embeddings?
- Embeddings are lists of numbers that represent text meaning
- Similar texts have similar embeddings
- We use them to find relevant documents for a query

Example:
    "dog" → [0.2, 0.5, 0.1, ...]
    "cat" → [0.3, 0.4, 0.1, ...]  # Similar to dog (both animals)
    "car" → [0.8, 0.1, 0.9, ...]  # Different (vehicle, not animal)
"""

from .embedder import Embedder

__all__ = ["Embedder"]
