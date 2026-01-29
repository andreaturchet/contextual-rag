"""
Vector Store Module
===================
This module handles storing and searching document embeddings.

What is a vector store?
- A database optimized for storing and searching vectors (embeddings)
- When you search, it finds vectors most similar to your query
- This is how we find relevant documents for a question

We use ChromaDB because:
- Easy to set up (no external server needed)
- Works well for small to medium datasets
- Can persist data to disk
"""

from .chroma_store import ChromaStore

__all__ = ["ChromaStore"]
