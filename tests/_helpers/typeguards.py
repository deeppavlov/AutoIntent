"""TypeGuard predicates used across test code to narrow union types from src/.

Each predicate has a runtime check that is cheap and a TypeGuard return type that
mypy uses to narrow the calling scope. Prefer these over `typing.cast` when the
narrowing has a verifiable runtime invariant the test relies on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from autointent.custom_types import ListOfGenericLabels, ListOfLabels


def is_strict_labels(labels: ListOfGenericLabels) -> TypeGuard[ListOfLabels]:
    """True iff every label is non-None (i.e. no OOS samples).

    Narrows `ListOfGenericLabels` (= `ListOfLabels | ListOfLabelsWithOOS`) to
    `ListOfLabels` so the call site can pass it to APIs that require the
    non-OOS variant (e.g. `Embedder.train`, `SentenceTransformerEmbeddingBackend.train`,
    `KNNScorer.fit`).
    """
    return all(label is not None for label in labels)
