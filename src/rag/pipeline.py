"""
RAG Pipeline
------------
Main module that combines retrieval and generation.

How it works:
1. User asks a question
2. Search for relevant documents (Retrieval)
3. Give documents + question to LLM (Augmented)
4. LLM generates answer (Generation)
"""

from typing import Dict, List
import time
import logging

from ..retrieval.retriever import Retriever
from ..retrieval.hf_reranker import HuggingFaceReranker

logger = logging.getLogger(__name__)

# RAG Prompt - optimized for direct, concise answers
RAG_PROMPT = """You are a helpful assistant for Infineon Technologies. Answer questions based ONLY on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer the question directly and concisely
- Focus on the CONTENT sections in the context (ignore KEY FACTS and QUESTIONS metadata)
- Do NOT start with phrases like "Let me analyze" or "Based on the documents"
- If the answer is not in the context, say "I don't have this information in the provided documents"
- Include specific facts, numbers, and names when available

ANSWER:"""


class RAGPipeline:
    """
    Main RAG pipeline for question answering.
    
    Usage:
        pipeline = RAGPipeline(embedder=embedder, vector_store=store, llm_client=client)
        result = pipeline.query("What are Infineon's products?")
        print(result["answer"])
    """
    
    def __init__(self, embedder, vector_store, llm_client, top_k: int = 5,
                 use_reranker: bool = True, prompt_template: str = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedder: Embedder instance
            vector_store: ChromaStore instance
            llm_client: OllamaClient instance
            top_k: Number of documents to retrieve
            use_reranker: Whether to use reranker (default: True)
            prompt_template: Custom prompt (uses optimized RAG_PROMPT by default)
        """
        self.llm_client = llm_client
        self.top_k = top_k
        self.prompt_template = prompt_template or RAG_PROMPT

        # Initialize retriever with optional reranker
        self._reranker = HuggingFaceReranker() if use_reranker else None
        self.retriever = Retriever(embedder=embedder, vector_store=vector_store, reranker=self._reranker)

    def unload(self):
        """Unload reranker model to free memory."""
        if self._reranker:
            self._reranker.unload()
            self._reranker = None

    def query(self, question: str) -> Dict:
        """
        Process a question and generate an answer.
        
        Args:
            question: The question

        Returns:
            Dictionary with answer, sources, num_docs, latency_seconds
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        docs = self.retriever.search(question, top_k=self.top_k)

        if not docs:
            return {
                "answer": "No relevant information found.",
                "sources": [],
                "num_docs": 0,
                "latency_seconds": time.time() - start_time
            }
        
        # Step 2: Build context from documents
        context = self._build_context(docs)

        # Step 3: Generate answer
        prompt = self.prompt_template.format(context=context, question=question)
        answer = self.llm_client.generate(prompt)
        
        # Step 4: Return result
        sources = list(set(doc.get("source", "") for doc in docs))

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "num_docs": len(docs),
            "latency_seconds": time.time() - start_time
        }
    
    def _build_context(self, documents: List[Dict]) -> str:
        """Build context string from documents."""
        parts = []
        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            source = doc.get("source", "Unknown")
            parts.append(f"[Document {i}] (Source: {source})\n{text}")
        
        return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    print("Testing RAG Pipeline")
    print("-" * 40)
    print("Requires Ollama and indexed documents")
