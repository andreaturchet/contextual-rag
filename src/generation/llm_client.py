"""
LLM Client
----------
Simple client for Ollama local LLM inference.

Why local LLMs:
- Data privacy: No data leaves the network
- No API fees
-inference costs
"""

import requests
from typing import List
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for Ollama API.

    Usage:
        client = OllamaClient(model="gemma3:4b")
        response = client.generate("What is AI?")
    """
    
    def __init__(
        self, 
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1
    ):
        """
        Initialize the client.
        
        Args:
            model: Model name (e.g., "gemma3:4b")
            base_url: Ollama server URL
            temperature: Randomness (0 = factual, 1 = creative)
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        logger.info(f"OllamaClient: model={model}")
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except:
            return []
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response.
        
        Args:
            prompt: The prompt

        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            return response.json().get("response", "")

        except requests.exceptions.ConnectionError:
            raise ConnectionError("Cannot connect to Ollama. Run: ollama serve")


if __name__ == "__main__":
    print("Testing Ollama Client")
    print("-" * 40)
    
    client = OllamaClient(model="gemma3:4b")
    
    if not client.is_available():
        print("Ollama is not running")
        print("Run: ollama serve")
        exit(1)
    
    print("Ollama is running")
    print(f"Models: {client.list_models()}")

    response = client.generate("What is a semiconductor? One sentence.")
    print(f"\nResponse: {response}")
