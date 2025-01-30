"""These modules are used only for optimization as they use proxy metrics for choosing best embedding model."""

from ._logreg import LogregAimedEmbedding
from ._retrieval import RetrievalAimedEmbedding

__all__ = ["LogregAimedEmbedding", "RetrievalAimedEmbedding"]
