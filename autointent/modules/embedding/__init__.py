"""These modules are used only for optimization as they use proxy metrics for choosing best embedding model."""

from ._retrieval import LogRegEmbedding, RetrievalEmbedding

__all__ = ["LogRegEmbedding", "RetrievalEmbedding"]
