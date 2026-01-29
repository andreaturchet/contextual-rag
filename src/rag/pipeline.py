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

logger = logging.getLogger(__name__)

# RAG Prompt - optimized based on research:
# - Chain-of-Thought prompting (Wei et al., 2022)
# - Citation for reducing hallucination (Gao et al., 2023)
RAG_PROMPT = """You are a technical assistant for Infineon Technologies, specializing in answering questions about company documents.

CONTEXT (Retrieved Documents):
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
Let's think step by step:
1. First, identify which documents are relevant to the question
2. Extract the key information from those documents
3. Formulate a concise answer based ONLY on the provided context
4. Cite the source document when making claims (e.g., "According to Document 1...")
5. If the context doesn't contain the answer, say "I don't have enough information in the provided documents"

YOUR ANSWER:"""


class RAGPipeline:
    """
    Main RAG pipeline for question answering.
    
    Usage:
        pipeline = RAGPipeline(retriever=retriever, llm_client=client)
        result = pipeline.query("What are Infineon's products?")
        print(result["answer"])
    """
    
    def __init__(self, retriever, llm_client, top_k: int = 5, prompt_template: str = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: Retriever instance
            llm_client: OllamaClient instance
            top_k: Number of documents to retrieve
            prompt_template: Custom prompt (uses optimized RAG_PROMPT by default)
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.prompt_template = prompt_template or RAG_PROMPT

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
