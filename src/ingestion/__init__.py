"""
Ingestion Module
================
This module handles loading and processing documents.

Components:
- DocumentLoader: Load files from various formats (PDF, TXT, MD)
- TextChunker: Split documents into smaller chunks
- ContextualChunker: Add context to chunks (advanced feature)
"""

from .document_loader import DocumentLoader
from .chunker import TextChunker

# Optional: uncomment when contextual chunker is ready
# from .contextual_chunker import ContextualChunker

__all__ = ["DocumentLoader", "TextChunker"]
