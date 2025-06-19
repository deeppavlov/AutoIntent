"""Some generative methods for enriching training datasets.

See :ref:`data-aug-tuts`.
"""

from ._generator import Generator, RetriesExceededError

__all__ = ["Generator", "RetriesExceededError"]
