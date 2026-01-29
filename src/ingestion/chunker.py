"""
Text Chunker
------------
Splits documents into smaller chunks for processing.

Why chunking:
- LLMs have token limits
- Smaller chunks give more precise retrieval
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text into overlapping chunks.
    
    Usage:
        chunker = TextChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk_documents(documents)
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Max characters per chunk
            overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text or len(text) <= self.chunk_size:
            return [text.strip()] if text else []

        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence or word boundary (only if not at end)
            if end < len(text):
                for sep in [". ", "\n", " "]:
                    pos = chunk.rfind(sep)
                    if pos > len(chunk) * 0.5:
                        chunk = chunk[:pos + 1]
                        break

            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Calculate advance: at least (chunk_size - overlap) to make progress
            # If remaining text is small, just finish
            if end >= len(text):
                # We've reached the end
                break

            # Normal case: advance by chunk length minus overlap
            advance = len(chunk) - self.overlap
            if advance < 1:
                advance = len(chunk)  # Don't overlap if chunk is too small
            start = start + advance

        return chunks
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """Chunk a document and keep metadata."""
        text = document.get("content", "")
        source = document.get("source", "unknown")

        text_chunks = self.chunk_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_index": i,
                "metadata": {"chunk_id": f"{source}_chunk_{i}"}
            })

        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk multiple documents."""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


if __name__ == "__main__":
    print("Testing Text Chunker")
    print("-" * 40)
    
    chunker = TextChunker(chunk_size=100, overlap=20)

    text = "This is a test. " * 20
    chunks = chunker.chunk_text(text)

    print(f"Original: {len(text)} chars")
    print(f"Chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0][:50]}...")
