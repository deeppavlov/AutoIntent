"""Embedder module with multiple backend support."""

from .base import BaseEmbeddingBackend
from .embedder import Embedder
from .hashing_vectorizer import HashingVectorizerEmbeddingBackend
from .openai import OpenaiEmbeddingBackend
from .sentence_transformers import SentenceTransformerEmbeddingBackend
from .vllm import VllmEmbeddingBackend

__all__ = [
    "BaseEmbeddingBackend",
    "Embedder",
    "HashingVectorizerEmbeddingBackend",
    "OpenaiEmbeddingBackend",
    "SentenceTransformerEmbeddingBackend",
    "VllmEmbeddingBackend",
]
