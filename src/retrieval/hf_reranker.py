"""
HuggingFace Reranker
--------------------
Uses Qwen3-Reranker to re-rank retrieved documents.

Reranking improves retrieval by using a more powerful model
to score query-document relevance.
"""

import logging
from typing import List, Dict
import torch

logger = logging.getLogger(__name__)


class HuggingFaceReranker:
    """
    Reranker using Qwen3-Reranker from HuggingFace.
    
    Usage:
        reranker = HuggingFaceReranker()
        reranked = reranker.rerank(query, documents, top_k=3)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = None
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name
            device: "cuda", "mps", or "cpu" (auto-detected if None)
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        # Prompt format for the model
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    def _load_model(self):
        """Load the model (lazy loading)."""
        if self._loaded:
            return
        
        print(f"  Loading reranker on {self.device}...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left',
            trust_remote_code=True
        )

        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Get token IDs for yes/no
        self.token_yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no = self.tokenizer.convert_tokens_to_ids("no")

        self._loaded = True
        print(f"  Reranker ready!")

    def unload(self):
        """Unload the model to free memory."""
        if self._loaded:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False

            # Force garbage collection
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Clear MPS cache on Apple Silicon
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()

            print("  Reranker unloaded")

    def compute_score(self, query: str, document: str) -> float:
        """Compute relevance score for a query-document pair."""
        self._load_model()
        
        # Format input
        text = f"<<Instruct>>: Given a query, retrieve relevant passages.\n<<Query>>: {query}\n<<Document>>: {document}"

        # Tokenize with prefix and suffix
        prefix_ids = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        content_ids = self.tokenizer.encode(text, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        input_ids = prefix_ids + content_ids + suffix_ids
        inputs = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits[:, -1, :]
            
            # Get probability of "yes"
            yes_score = logits[0, self.token_yes].item()
            no_score = logits[0, self.token_no].item()

            import math
            max_score = max(yes_score, no_score)
            yes_exp = math.exp(yes_score - max_score)
            no_exp = math.exp(no_score - max_score)

            return yes_exp / (yes_exp + no_exp)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-rank documents by relevance.

        Args:
            query: The search query
            documents: List of document dicts with "text" field
            top_k: Number of top documents to return

        Returns:
            Re-ranked documents (modified in-place with rerank_score)
        """
        if not documents:
            return []
        
        self._load_model()
        
        # Score each document IN-PLACE to save memory
        for doc in documents:
            text = doc.get("text", "")
            doc["rerank_score"] = self.compute_score(query, text)

        # Sort by score (highest first)
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)

        return documents[:top_k]


def test_reranker():
    """Test the reranker."""
    print("=" * 50)
    print("Testing Qwen3 Reranker")
    print("=" * 50)
    
    print("\nInitializing (downloads model on first run)...")
    reranker = HuggingFaceReranker()
    
    query = "What products does Infineon make?"
    
    documents = [
        {"text": "Infineon Technologies AG is headquartered in Neubiberg, Germany.", "source": "doc1"},
        {"text": "Infineon makes power semiconductors, microcontrollers, and sensors for automotive applications.", "source": "doc2"},
        {"text": "The company was founded in 1999 as a spin-off from Siemens.", "source": "doc3"},
        {"text": "Infineon produces IGBT modules and power MOSFETs for electric vehicles.", "source": "doc4"},
    ]
    
    print(f"\nQuery: {query}")
    print(f"Documents: {len(documents)}")
    
    print("\nComputing scores (this may take a while on CPU)...")
    import time
    start = time.time()
    
    reranked = reranker.rerank(query, documents, top_k=4)
    
    elapsed = time.time() - start
    print(f"\nTime: {elapsed:.2f}s ({elapsed/len(documents):.2f}s per doc)")
    
    print("\nReranked results:")
    for i, doc in enumerate(reranked, 1):
        print(f"  {i}. Score: {doc['rerank_score']:.4f} - {doc['text'][:50]}...")
    
    print("\n" + "=" * 50)
    return reranked


if __name__ == "__main__":
    test_reranker()
