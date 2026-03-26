"""
Contextual Chunker
-------------------------------------
Adds context to chunks using an LLM.

Based on Anthropic's Contextual Retrieval technique:
https://www.anthropic.com/engineering/contextual-retrieval

MEMORY OPTIMIZATIONS:
- Removed redundant 'original_text' field (already stored in 'text')
- Aggressive garbage collection every 2 chunks
- Explicit deletion of response objects
- Truncated document stored once, not per chunk

RETRIEVAL OPTIMIZATIONS (v2):
- Improved prompt with specific rules
- Extracts KEY FACTS not descriptions
- Hybrid format: context + questions + content
"""

from typing import List, Dict
import logging
import gc

logger = logging.getLogger(__name__)


class ContextualChunker:
    """
    Memory-optimized contextual chunker with improved retrieval.

    Optimized for systems with limited RAM (16GB or less).
    """
    
    # Lightweight prompt - only KEY_FACTS for faster processing
    PROMPT = """<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Extract 3-5 KEY FACTS from this chunk (names, numbers, definitions, dates). Be SPECIFIC - include actual values.

Format:
KEY_FACTS:
- fact 1
- fact 2
- fact 3"""

    def __init__(self, llm_client, max_doc_length: int = 5000, batch_size: int = 2):
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

        # Parse response to extract facts
        key_facts = ""

        if "KEY_FACTS:" in context:
            key_facts = context.replace("KEY_FACTS:", "").strip()
        else:
            # Fallback: use entire response as context
            key_facts = context

        # Store parsed context
        chunk["context"] = context
        chunk["key_facts"] = key_facts

        # Build contextualized text: Source -> Key facts -> Original content
        contextualized_parts = []

        # Add document source if available
        source = chunk.get("source", "")
        if source:
            # Extract filename from path
            import os
            filename = os.path.basename(source).replace("_", " ").replace(".txt", "")
            contextualized_parts.append(f"SOURCE: {filename}")

        if key_facts:
            contextualized_parts.append(f"KEY FACTS:\n{key_facts}")

        contextualized_parts.append(f"CONTENT:\n{chunk_text}")

        chunk["contextualized_text"] = "\n\n".join(contextualized_parts)

        # Clear local variables
        del context, key_facts


if __name__ == "__main__":
    print("Contextual Chunker - Memory Optimized v2")
    print("-" * 40)
    print("Based on Anthropic's Contextual Retrieval")
    print("With question-aware context generation")
    print("Optimized for 16GB RAM systems")
    print("Run from main.py or demo.py to test")
