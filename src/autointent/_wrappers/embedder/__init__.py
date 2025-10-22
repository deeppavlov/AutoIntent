"""Embedder module with multiple backend support."""

from .base import BaseEmbeddingBackend
from .embedder import Embedder
from .openai import OpenaiEmbeddingBackend
from .sentence_transformers import SentenceTransformerEmbeddingBackend

__all__ = [
    "BaseEmbeddingBackend",
    "Embedder",
    "OpenaiEmbeddingBackend",
    "SentenceTransformerEmbeddingBackend",
]
