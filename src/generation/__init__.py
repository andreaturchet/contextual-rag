"""
Generation Module
=================
This module handles text generation using local LLMs.

Components:
- OllamaClient: Interface to Ollama for local LLM inference
"""

from .llm_client import OllamaClient

__all__ = ["OllamaClient"]
