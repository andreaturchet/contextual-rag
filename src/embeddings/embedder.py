"""
Embedder stuff logic
----------------------------
Generates embeddings for text.

Embeddings convert text to numbers so computers can compare meaning.
Similar texts have similar vectors.

MEMORY OPTIMIZATIONS:
- Uses Ollama's native batch API (sends multiple texts at once)
- Explicit response cleanup
- Small batch sizes for memory-constrained systems
"""

import requests
import logging
import gc
from typing import List

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generate embeddings using Ollama with memory optimization.

    Usage:
        embedder = Embedder()
        vector = embedder.embed("What is machine learning?")
    """
    
    def __init__(
        self,
        model_name: str = "qwen3-embedding:0.6b",
        ollama_url: str = "http://localhost:11434",
        provider: str = None  # kept for compatibility, not used
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: Ollama embedding model name
            ollama_url: URL for Ollama server
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        logger.info(f"Embedder initialized: model={self.model_name}")

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: The text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        url = f"{self.ollama_url}/api/embed"
        payload = {
            "model": self.model_name,
            "input": text
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            embeddings = result.get("embeddings", [])

            # Cleanup
            del response
            del result

            if embeddings:
                embedding = embeddings[0]
                del embeddings
                return embedding

            raise ValueError("No embedding returned")

        except requests.exceptions.ConnectionError:
            raise ConnectionError("Cannot connect to Ollama. Run: ollama serve")

    def embed_batch(self, texts: List[str], batch_size: int = 2) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with memory-efficient batching.

        Uses Ollama's native batch API when possible.

        Args:
            texts: List of texts
            batch_size: Number of texts per API call (small for memory)

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            try:
                # Try native batch API
                batch_embeddings = self._embed_batch_native(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                # Fallback to one-by-one
                logger.warning(f"Batch API failed, falling back: {e}")
                for text in batch_texts:
                    all_embeddings.append(self.embed(text))

            # Cleanup after each batch
            gc.collect()

        return all_embeddings

    def _embed_batch_native(self, texts: List[str]) -> List[List[float]]:
        """Use Ollama's native batch embedding API."""
        url = f"{self.ollama_url}/api/embed"
        payload = {
            "model": self.model_name,
            "input": texts  # Ollama accepts list of strings
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        embeddings = result.get("embeddings", [])

        # Cleanup
        del response
        del result
        gc.collect()

        return embeddings


if __name__ == "__main__":
    print("Testing Embedder")
    print("-" * 40)
    
    try:
        embedder = Embedder()
        text = "What is machine learning?"
        embedding = embedder.embed(text)
        
        print(f"Text: {text}")
        print(f"Dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print("\nTest passed!")

    except ConnectionError:
        print("Ollama not running. Start with: ollama serve")
