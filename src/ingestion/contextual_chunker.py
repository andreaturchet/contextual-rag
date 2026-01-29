"""
Contextual Chunker - Memory Optimized
-------------------------------------
Adds context to chunks using an LLM.

Based on Anthropic's Contextual Retrieval technique:
https://www.anthropic.com/engineering/contextual-retrieval

MEMORY OPTIMIZATIONS:
- Removed redundant 'original_text' field (already stored in 'text')
- Aggressive garbage collection every 2 chunks
- Explicit deletion of response objects
- Truncated document stored once, not per chunk
"""

from typing import List, Dict
import logging
import gc

logger = logging.getLogger(__name__)


class ContextualChunker:
    """
    Memory-optimized contextual chunker.

    Optimized for systems with limited RAM (16GB or less).
    """
    
    # Shorter prompt template to reduce memory
    PROMPT = """<document>
{document}
</document>

Chunk to contextualize:
<chunk>
{chunk}
</chunk>

Give a short context (1-2 sentences) to situate this chunk within the document for search retrieval. Answer only with the context."""

    def __init__(self, llm_client, max_doc_length: int = 3000, batch_size: int = 2):
        """
        Initialize the contextual chunker.
        
        Args:
            llm_client: LLM client with generate() method
            max_doc_length: Max characters of document to include (reduced for memory)
            batch_size: Chunks to process before garbage collection
        """
        self.llm_client = llm_client
        self.max_doc_length = max_doc_length
        self.batch_size = batch_size

    def add_context_to_chunks(self, chunks: List[Dict], document_text: str) -> List[Dict]:
        """
        Add context to chunks IN-PLACE with aggressive memory management.

        Args:
            chunks: List of chunk dictionaries
            document_text: Full source document text

        Returns:
            Same chunks list with 'contextualized_text' field added
        """
        # Truncate document ONCE and store
        if len(document_text) > self.max_doc_length:
            doc_text = document_text[:self.max_doc_length] + "\n[...]"
        else:
            doc_text = document_text

        # Clear original document reference
        del document_text
        gc.collect()

        total = len(chunks)

        for i, chunk in enumerate(chunks):
            try:
                self._add_context_inplace(chunk, doc_text)
            except Exception as e:
                logger.warning(f"Failed chunk {i}: {e}")
                chunk["contextualized_text"] = chunk["text"]

            # Aggressive cleanup every batch_size chunks
            if (i + 1) % self.batch_size == 0:
                gc.collect()
                print(f"  Contextualized {i + 1}/{total} chunks...")

        # Final cleanup
        gc.collect()
        return chunks

    def _add_context_inplace(self, chunk: Dict, document_text: str) -> None:
        """Add context to a single chunk with minimal memory allocation."""
        chunk_text = chunk.get("text", "")
        
        # Build prompt
        prompt = self.PROMPT.format(document=document_text, chunk=chunk_text)

        # Get context from LLM
        response = self.llm_client.generate(prompt)
        context = response.strip() if response else ""

        # Explicitly release response
        del response
        del prompt

        # Store context and build contextualized_text
        # NOTE: We don't store 'original_text' - it's redundant with 'text'
        chunk["context"] = context
        chunk["contextualized_text"] = f"CONTEXT: {context}\n\nCONTENT: {chunk_text}"

        # Clear local variable
        del context


if __name__ == "__main__":
    print("Contextual Chunker - Memory Optimized")
    print("-" * 40)
    print("Based on Anthropic's Contextual Retrieval")
    print("Optimized for 16GB RAM systems")
    print("Run from main.py or demo.py to test")
