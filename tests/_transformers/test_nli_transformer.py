from __future__ import annotations

import importlib.resources as ires
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from autointent import Dataset, Ranker
from autointent.configs import CrossEncoderConfig
from autointent.context.data_handler import DataHandler

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from autointent.custom_types import ListOfGenericLabels, ListOfLabels, RerankedItem

pytest.importorskip("sentence_transformers")


@pytest.fixture
def data_handler() -> DataHandler:
    data_path = cast("Path", ires.files("tests.assets.data").joinpath("clinc_subset.json"))
    return DataHandler(dataset=Dataset.from_json(data_path), random_seed=42)


def test_nli_transformer_predict_without_trained_head(data_handler: DataHandler) -> None:
    model = Ranker(cross_encoder_config={"model_name": "cross-encoder/ms-marco-MiniLM-L6-v2", "train_head": True})
    with pytest.raises(ValueError, match="Classifier is not trained yet"):
        model.predict(data_handler.train_utterances(0))  # type: ignore[arg-type]  # reason: intentionally passing list[str] (not list[tuple[str, str]]) to trigger the not-trained-yet path before any type-narrowed call happens


def build_pairs(texts: list[str]) -> list[tuple[str, str]]:
    return [(texts[0], t) for t in texts[1:]]


def check_predictions(predicted: npt.NDArray[np.float32], labels: ListOfGenericLabels) -> None:
    # we compare first text with others, hence we expect that minimum prob in the class is bigger max prob out the class
    min_in_class = float("inf")
    max_out_class = float("-inf")
    ref_label = labels[0]
    for prob, lbl in zip(predicted, labels[1:], strict=False):
        if ref_label == lbl:
            min_in_class = min(min_in_class, float(prob))
        else:
            max_out_class = max(max_out_class, float(prob))
    assert min_in_class > max_out_class


def check_ranking(ranked: list[RerankedItem], labels: ListOfGenericLabels) -> None:
    for r in ranked:
        assert hasattr(r, "corpus_id")
        assert hasattr(r, "score")

    # We expect the same label text to go first, then the rest
    ranked_labels = [labels[r.corpus_id + 1] == labels[0] for r in ranked]
    expected_labels = [labels[0] == lbl for lbl in labels[1:]]
    expected_labels.sort(reverse=True)
    assert ranked_labels == expected_labels


def test_nli_transformer_predict_with_train_head(data_handler: DataHandler) -> None:
    model = Ranker(cross_encoder_config={"model_name": "cross-encoder/ms-marco-MiniLM-L6-v2", "train_head": True})
    texts = data_handler.train_utterances(0)
    labels = data_handler.train_labels(0)
    # clinc_subset has no OOS samples, so labels is statically ListOfLabels at runtime.
    model.fit(texts, cast("ListOfLabels", labels))
    predicted = model.predict(build_pairs(texts))
    check_predictions(predicted, labels)

    ranked = model.rank(texts[0], texts[1:])
    check_ranking(ranked, labels)


def test_nli_transformer_predict_default(data_handler: DataHandler) -> None:
    model = Ranker(cross_encoder_config={"model_name": "cross-encoder/ms-marco-MiniLM-L6-v2", "train_head": False})
    texts = data_handler.train_utterances(0)
    labels = data_handler.train_labels(0)
    predicted = model.predict(build_pairs(texts))
    check_predictions(predicted, labels)

    ranked = model.rank(texts[0], texts[1:])
    check_ranking(ranked, labels)


def test_nli_transformer_predict_default_with_fit(data_handler: DataHandler) -> None:
    model = Ranker(cross_encoder_config={"model_name": "cross-encoder/ms-marco-MiniLM-L6-v2", "train_head": False})
    texts = data_handler.train_utterances(0)
    labels = data_handler.train_labels(0)
    # clinc_subset has no OOS samples, so labels is statically ListOfLabels at runtime.
    model.fit(texts, cast("ListOfLabels", labels))
    predicted = model.predict(build_pairs(texts))
    check_predictions(predicted, labels)

    ranked = model.rank(texts[0], texts[1:])
    check_ranking(ranked, labels)


def test_ranker_passes_revision_to_cross_encoder() -> None:
    # CrossEncoderConfig defaults to ms-marco-MiniLM-L6-v2; the validator
    # fills `revision` from DEFAULT_REVISIONS automatically.
    cfg = CrossEncoderConfig()
    assert cfg.revision is not None  # sanity: validator did its job

    with patch("sentence_transformers.CrossEncoder") as mock_ce:
        mock_ce.return_value = MagicMock()
        Ranker(cross_encoder_config=cfg)

    _, kwargs = mock_ce.call_args
    assert kwargs.get("revision") == cfg.revision, (
        f"Ranker must forward revision={cfg.revision!r} to CrossEncoder(); actual kwargs={kwargs}"
    )
