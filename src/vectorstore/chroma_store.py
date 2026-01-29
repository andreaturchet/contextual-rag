"""
ChromaDB Vector Store
---------------------
Stores document embeddings and enables similarity search.

ChromaDB is a local vector database that:
- Stores text and embeddings
- Enables fast similarity search
- Persists data to disk
"""

from typing import List, Dict
import os
import logging

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    Vector store using ChromaDB.
    
    Usage:
        store = ChromaStore(persist_directory="./data/vectordb")
        store.add_documents(chunks, embeddings)
        results = store.search(query_embedding, top_k=5)
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/vectordb",
        collection_name: str = "documents"
    ):
        """
        Initialize ChromaDB store.
        
        Args:
            persist_directory: Where to save the database
            collection_name: Name for the collection
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError("Install: pip install chromadb")
        
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        logger.info(f"ChromaDB ready: {self.collection.count()} documents")
    
    def add_documents(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        """
        Add documents to the store - MEMORY OPTIMIZED.

        For Contextual Retrieval: embeddings should be generated from
        contextualized_text (with CONTEXT prefix), but we store both
        versions for retrieval vs display.

        Args:
            chunks: List of chunk dictionaries
            embeddings: Corresponding embedding vectors
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match embeddings")
        
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("metadata", {}).get("chunk_id", f"chunk_{i}")
            ids.append(chunk_id)

            # Use contextualized_text if available (for Contextual Retrieval)
            # This is what gets stored and searched
            text = chunk.get("contextualized_text", chunk.get("text", ""))
            documents.append(text)

            # MEMORY OPTIMIZATION: Don't store original_text in metadata
            # We can extract it from the document if needed (after CONTENT:)
            metadatas.append({
                "source": str(chunk.get("source", "unknown")),
                "chunk_index": chunk.get("chunk_index", i),
            })

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        # Don't print for single-document adds (too noisy)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            
        Returns:
            List of documents with text, source, score
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                documents.append({
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i].get("source", "unknown"),
                    "score": 1 - results["distances"][0][i]
                })

        return documents
    
    def count(self) -> int:
        """Get number of documents in store."""
        return self.collection.count()


if __name__ == "__main__":
    print("Testing ChromaDB Store")
    print("-" * 40)
    
    import tempfile
    import random
    
    temp_dir = tempfile.mkdtemp()
    store = ChromaStore(persist_directory=temp_dir, collection_name="test")
    
    # Test data
    chunks = [
        {"text": "Infineon is a semiconductor company.",
         "source": "test.txt", "metadata": {"chunk_id": "0"}},
        {"text": "The company makes chips for cars.", 
         "source": "test.txt", "metadata": {"chunk_id": "1"}},
    ]
    
    # Fake embeddings
    embeddings = [[random.random() for _ in range(384)] for _ in chunks]

    store.add_documents(chunks, embeddings)
    print(f"Documents: {store.count()}")

    results = store.search(embeddings[0], top_k=2)
    print(f"Search results: {len(results)}")
    
    import shutil
    shutil.rmtree(temp_dir)
    print("Test completed")
