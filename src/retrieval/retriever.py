"""
Retriever
---------
Searches for relevant documents given a query.
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant documents for a query.
    
    Usage:
        retriever = Retriever(embedder=embedder, vector_store=store)
        results = retriever.search("What is machine learning?")
    """
    
    def __init__(
        self,
        embedder,
        vector_store,
        top_k: int = 5,
        reranker=None,
        rerank_top_k: int = None,
        score_threshold: float = 0.0  # kept for compatibility
    ):
        """
        Initialize the retriever.
        
        Args:
            embedder: Embedder instance
            vector_store: ChromaStore instance
            top_k: Number of documents to retrieve
            reranker: Optional reranker instance
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k or top_k

    def search(self, query: str, top_k: Optional[int] = None, use_reranker: bool = True) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of documents with text, source, score
        """
        if top_k is None:
            top_k = self.top_k
        
        # Convert query to embedding
        query_embedding = self.embedder.embed(query)
        
        # Get more docs if using reranker
        initial_top_k = top_k * 2 if self.reranker and use_reranker else top_k

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=initial_top_k)
        
        # Apply reranker if available
        if self.reranker and use_reranker and results:
            results = self.reranker.rerank(query, results, top_k=self.rerank_top_k)
        
        return results[:top_k]

    def count_documents(self) -> int:
        """Get total number of indexed documents."""
        return self.vector_store.count()


if __name__ == "__main__":
    print("Testing Retriever")
    print("-" * 40)
    print("Requires Ollama and indexed documents")
