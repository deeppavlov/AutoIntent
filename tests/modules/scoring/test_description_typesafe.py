from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from autointent import Context
from autointent.configs import DataConfig
from autointent.context.data_handler import DataHandler
from autointent.modules.scoring import TypeSafeDescriptionScorer
from autointent.modules.scoring._description.typesafe import (
    CHOICE_KEY,
    MAX_CHOICE_OPTIONS,
    build_questions,
    parse_answers,
)
from tests._fixtures.mock_typesafe import NOUL_NO, NOUL_YES, expected_best_index, fake_response
from tests._helpers import is_strict_labels

if TYPE_CHECKING:
    from pathlib import Path

    from autointent import Dataset
    from autointent.modules.scoring._description.typesafe import QuestionType
    from tests._fixtures.mock_typesafe import FakeAsyncTypeSafeClient, FakeTypeSafeClient

TEST_UTTERANCES = ["What is the balance on my account?", "How do I reset my online banking password?"]


def _descriptions(data_handler: DataHandler) -> list[str]:
    descriptions = data_handler.intent_descriptions
    assert all(d is not None for d in descriptions)
    return cast("list[str]", descriptions)


def test_build_questions_choice_and_noul() -> None:
    descriptions = ["book a hotel", "check the weather"]
    choice = build_questions("choice", descriptions)
    assert list(choice) == [CHOICE_KEY]
    assert choice[CHOICE_KEY]["type"] == "choice"
    assert choice[CHOICE_KEY]["criteria"] == {"intent_0": "book a hotel", "intent_1": "check the weather"}

    noul = build_questions("noul", descriptions)
    assert list(noul) == ["intent_0", "intent_1"]
    assert all(
        q["type"] == "noul" and desc in q["instructions"] for q, desc in zip(noul.values(), descriptions, strict=True)
    )


def test_parse_answers_orders_by_index() -> None:
    questions = build_questions("choice", ["a", "b", "c"])
    response = fake_response(questions, "x", best_index=2)
    assert parse_answers("choice", response.answers, 3) == pytest.approx([0.15, 0.15, 0.7])

    questions = build_questions("noul", ["a", "b", "c"])
    response = fake_response(questions, "x", best_index=1)
    assert parse_answers("noul", response.answers, 3) == pytest.approx([0.1, 0.9, 0.1])


def test_choice_multiclass_reproduces_model_distribution_at_unit_temperature(
    dataset: Dataset, patch_typesafe_scorer_client: tuple[FakeTypeSafeClient, FakeAsyncTypeSafeClient]
) -> None:
    sync_client, _ = patch_typesafe_scorer_client
    data_handler = DataHandler(dataset)
    descriptions = _descriptions(data_handler)

    labels = data_handler.train_labels(0)
    assert is_strict_labels(labels)

    scorer = TypeSafeDescriptionScorer(question_type="choice", temperature=1.0, max_concurrent=None)
    scorer.fit(data_handler.train_utterances(0), labels, descriptions)
    assert scorer._description_texts == descriptions

    probabilities = scorer.predict(TEST_UTTERANCES)

    assert probabilities.shape == (len(TEST_UTTERANCES), len(descriptions))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    expected_response = fake_response(build_questions("choice", descriptions), TEST_UTTERANCES[0])
    expected = expected_response.answers[CHOICE_KEY].probabilities
    np.testing.assert_allclose(probabilities[0], [expected[f"intent_{i}"] for i in range(len(descriptions))], atol=1e-5)
    assert sync_client.calls == TEST_UTTERANCES


def test_noul_multilabel_reproduces_model_probabilities_at_unit_temperature(
    dataset: Dataset, patch_typesafe_scorer_client: tuple[FakeTypeSafeClient, FakeAsyncTypeSafeClient]
) -> None:
    data_handler = DataHandler(dataset.to_multilabel())
    descriptions = _descriptions(data_handler)

    labels = data_handler.train_labels(0)
    assert is_strict_labels(labels)

    scorer = TypeSafeDescriptionScorer(question_type="noul", temperature=1.0, max_concurrent=None, multilabel=True)
    scorer.fit(data_handler.train_utterances(0), labels, descriptions)

    probabilities = scorer.predict(TEST_UTTERANCES)

    assert probabilities.shape == (len(TEST_UTTERANCES), len(descriptions))
    winner = expected_best_index(TEST_UTTERANCES[0], len(descriptions))
    expected = [NOUL_YES if i == winner else NOUL_NO for i in range(len(descriptions))]
    np.testing.assert_allclose(probabilities[0], expected, atol=1e-5)


def test_temperature_sharpens_choice_distribution(
    dataset: Dataset, patch_typesafe_scorer_client: tuple[FakeTypeSafeClient, FakeAsyncTypeSafeClient]
) -> None:
    data_handler = DataHandler(dataset)
    descriptions = _descriptions(data_handler)
    cold = TypeSafeDescriptionScorer(question_type="choice", temperature=0.5, max_concurrent=None)
    cold.fit([], [], descriptions)
    probabilities = cold.predict(TEST_UTTERANCES[:1])
    winner = expected_best_index(TEST_UTTERANCES[0], len(descriptions))
    assert probabilities[0, winner] > 0.7  # softmax(log p / 0.5) sharpens the 0.7 winner


def test_choice_with_multilabel_is_rejected() -> None:
    with pytest.raises(ValueError, match="noul"):
        TypeSafeDescriptionScorer(question_type="choice", multilabel=True)


def test_from_context_defaults_question_type_by_task(dataset: Dataset) -> None:
    context = Context()
    context.set_dataset(dataset, DataConfig(scheme="ho"))
    assert TypeSafeDescriptionScorer.from_context(context).question_type == "choice"

    multilabel_context = Context()
    multilabel_context.set_dataset(dataset.to_multilabel(), DataConfig(scheme="ho"))
    scorer = TypeSafeDescriptionScorer.from_context(multilabel_context)
    assert scorer.question_type == "noul"
    assert scorer.get_implicit_initialization_params() == {"multilabel": True, "question_type": "noul"}


def test_from_context_coerces_hpo_sampled_choice_to_noul_on_multilabel(
    dataset: Dataset, caplog: pytest.LogCaptureFixture
) -> None:
    """An HPO trial can sample `question_type="choice"` from the preset's search space even on a
    multilabel dataset; `from_context` must downgrade it to `noul` with a warning instead of
    raising, so a single bad trial does not abort `Pipeline.fit()`.
    """
    multilabel_context = Context()
    multilabel_context.set_dataset(dataset.to_multilabel(), DataConfig(scheme="ho"))

    with caplog.at_level("WARNING"):
        scorer = TypeSafeDescriptionScorer.from_context(multilabel_context, question_type="choice")

    assert scorer.question_type == "noul"
    assert "question_type='choice'" in caplog.text
    assert scorer.get_implicit_initialization_params() == {"multilabel": True, "question_type": "noul"}


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        TypeSafeDescriptionScorer(max_concurrent=None).predict(["hello"])


def test_async_path_uses_async_client_and_matches_sync(
    dataset: Dataset, patch_typesafe_scorer_client: tuple[FakeTypeSafeClient, FakeAsyncTypeSafeClient]
) -> None:
    sync_client, async_client = patch_typesafe_scorer_client
    descriptions = _descriptions(DataHandler(dataset))

    concurrent = TypeSafeDescriptionScorer(question_type="choice", max_concurrent=2, max_per_second=100)
    concurrent.fit([], [], descriptions)
    concurrent_probabilities = concurrent.predict(TEST_UTTERANCES)
    assert async_client.calls == TEST_UTTERANCES
    assert sync_client.calls == []

    sequential = TypeSafeDescriptionScorer(question_type="choice", max_concurrent=None, use_cache=False)
    sequential.fit([], [], descriptions)
    sequential_probabilities = sequential.predict(TEST_UTTERANCES)
    np.testing.assert_allclose(sequential_probabilities, concurrent_probabilities)

    # A swapped result order between utterances would slip past assert_allclose above if it
    # happened to swap two utterances with identical distributions; check per-utterance winners.
    expected_indices = [expected_best_index(utterance, len(descriptions)) for utterance in TEST_UTTERANCES]
    np.testing.assert_array_equal(np.argmax(concurrent_probabilities, axis=1), expected_indices)
    np.testing.assert_array_equal(np.argmax(sequential_probabilities, axis=1), expected_indices)


@pytest.mark.parametrize("max_concurrent", [None, 2])
@pytest.mark.parametrize("question_type", ["choice", "noul"])
def test_failed_request_yields_uniform_row_and_warns(
    dataset: Dataset,
    patch_typesafe_scorer_client: tuple[FakeTypeSafeClient, FakeAsyncTypeSafeClient],
    question_type: QuestionType,
    max_concurrent: int | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The uniform fallback is exercised for both the softmax (choice) and sigmoid (noul) paths."""
    sync_client, async_client = patch_typesafe_scorer_client
    descriptions = _descriptions(DataHandler(dataset))

    def boom(*_: object, **__: object) -> None:
        msg = "simulated outage"
        raise RuntimeError(msg)

    async def boom_async(*_: object, **__: object) -> None:
        boom()

    sync_client.system_one = boom  # type: ignore[assignment]
    async_client.system_one = boom_async  # type: ignore[assignment]

    multilabel = question_type == "noul"
    scorer = TypeSafeDescriptionScorer(
        question_type=question_type, max_concurrent=max_concurrent, multilabel=multilabel
    )
    scorer.fit([], [], descriptions)
    with caplog.at_level("WARNING"):
        probabilities = scorer.predict(TEST_UTTERANCES)

    np.testing.assert_allclose(probabilities, 1.0 / len(descriptions))
    assert "simulated outage" in caplog.text


class FakeAPIError(Exception):
    """Stand-in for typesafe_sdk's HTTP error types, which carry `.status` (see `_is_fatal`)."""

    def __init__(self, status: int) -> None:
        super().__init__(f"api error, status={status}")
        self.status = status


@pytest.mark.parametrize("max_concurrent", [None, 2])
def test_fatal_status_propagates_but_transient_status_falls_back(
    dataset: Dataset,
    patch_typesafe_scorer_client: tuple[FakeTypeSafeClient, FakeAsyncTypeSafeClient],
    max_concurrent: int | None,
) -> None:
    """A 401 (bad key/misconfiguration) must not be swallowed into a uniform row; a 503 still is."""
    sync_client, async_client = patch_typesafe_scorer_client
    descriptions = _descriptions(DataHandler(dataset))

    def boom_401(*_: object, **__: object) -> None:
        raise FakeAPIError(401)

    async def boom_401_async(*_: object, **__: object) -> None:
        boom_401()

    sync_client.system_one = boom_401  # type: ignore[assignment]
    async_client.system_one = boom_401_async  # type: ignore[assignment]

    fatal_scorer = TypeSafeDescriptionScorer(question_type="choice", max_concurrent=max_concurrent)
    fatal_scorer.fit([], [], descriptions)
    with pytest.raises(FakeAPIError):
        fatal_scorer.predict(TEST_UTTERANCES)

    def boom_503(*_: object, **__: object) -> None:
        raise FakeAPIError(503)

    async def boom_503_async(*_: object, **__: object) -> None:
        boom_503()

    sync_client.system_one = boom_503  # type: ignore[assignment]
    async_client.system_one = boom_503_async  # type: ignore[assignment]

    transient_scorer = TypeSafeDescriptionScorer(question_type="choice", max_concurrent=max_concurrent)
    transient_scorer.fit([], [], descriptions)
    probabilities = transient_scorer.predict(TEST_UTTERANCES)
    np.testing.assert_allclose(probabilities, 1.0 / len(descriptions))


def test_fit_rejects_too_many_choice_options() -> None:
    descriptions = [f"description {i}" for i in range(MAX_CHOICE_OPTIONS + 1)]
    scorer = TypeSafeDescriptionScorer(question_type="choice")
    with pytest.raises(ValueError, match="noul"):
        scorer.fit([], [], descriptions)


def test_cache_hit_skips_the_client(
    dataset: Dataset, patch_typesafe_scorer_client: tuple[FakeTypeSafeClient, FakeAsyncTypeSafeClient]
) -> None:
    sync_client, _ = patch_typesafe_scorer_client
    descriptions = _descriptions(DataHandler(dataset))

    scorer = TypeSafeDescriptionScorer(question_type="choice", max_concurrent=None)
    scorer.fit([], [], descriptions)
    first = scorer.predict(TEST_UTTERANCES)
    assert len(sync_client.calls) == len(TEST_UTTERANCES)

    second = scorer.predict(TEST_UTTERANCES)
    assert len(sync_client.calls) == len(TEST_UTTERANCES)
    np.testing.assert_allclose(first, second)

    # A fresh instance (new HPO trial) with the same descriptions also hits the disk cache.
    other = TypeSafeDescriptionScorer(question_type="choice", max_concurrent=None, temperature=2.0)
    other.fit([], [], descriptions)
    other.predict(TEST_UTTERANCES)
    assert len(sync_client.calls) == len(TEST_UTTERANCES)

    # A different question_type is a different key.
    noul = TypeSafeDescriptionScorer(question_type="noul", max_concurrent=None)
    noul.fit([], [], descriptions)
    noul.predict(TEST_UTTERANCES)
    assert len(sync_client.calls) == 2 * len(TEST_UTTERANCES)


def test_cache_key_changes_with_model(
    dataset: Dataset, patch_typesafe_scorer_client: tuple[FakeTypeSafeClient, FakeAsyncTypeSafeClient]
) -> None:
    """A different `model` must be a different cache key, else a cached answer from one model would
    be served to another (the model determines the answer, so the key must cover it).
    """
    sync_client, _ = patch_typesafe_scorer_client
    descriptions = _descriptions(DataHandler(dataset))

    scorer = TypeSafeDescriptionScorer(question_type="choice", max_concurrent=None)
    scorer.fit([], [], descriptions)
    scorer.predict(TEST_UTTERANCES)
    assert len(sync_client.calls) == len(TEST_UTTERANCES)

    other_model = TypeSafeDescriptionScorer(question_type="choice", max_concurrent=None, model="jev-preview")
    other_model.fit([], [], descriptions)
    other_model.predict(TEST_UTTERANCES)
    assert len(sync_client.calls) == 2 * len(TEST_UTTERANCES)


def test_predict_logs_usage(
    dataset: Dataset,
    patch_typesafe_scorer_client: tuple[FakeTypeSafeClient, FakeAsyncTypeSafeClient],
    caplog: pytest.LogCaptureFixture,
) -> None:
    descriptions = _descriptions(DataHandler(dataset))
    scorer = TypeSafeDescriptionScorer(question_type="choice", max_concurrent=None)
    scorer.fit([], [], descriptions)
    with caplog.at_level("INFO", logger="autointent.modules.scoring._description.typesafe"):
        scorer.predict(TEST_UTTERANCES)
    assert "2 requests" in caplog.text
    assert "200 input tokens" in caplog.text
    assert "model jev-test" in caplog.text


@pytest.mark.parametrize("max_concurrent", [None, 2])
def test_dump_and_load_roundtrip(
    dataset: Dataset,
    patch_typesafe_scorer_client: tuple[FakeTypeSafeClient, FakeAsyncTypeSafeClient],
    tmp_path: Path,
    max_concurrent: int | None,
) -> None:
    descriptions = _descriptions(DataHandler(dataset))
    scorer = TypeSafeDescriptionScorer(
        question_type="noul",
        model="jev-preview",
        temperature=0.7,
        max_concurrent=max_concurrent,
        max_retries=5,
        use_cache=False,
    )
    scorer.fit([], [], descriptions)
    before = scorer.predict(TEST_UTTERANCES)

    dump_dir = tmp_path / "dump"
    scorer.dump(str(dump_dir))
    loaded = TypeSafeDescriptionScorer.load(str(dump_dir))

    assert loaded.question_type == "noul"
    assert loaded.model == "jev-preview"
    assert loaded.temperature == pytest.approx(0.7)
    assert loaded.max_concurrent == max_concurrent
    assert loaded.max_retries == 5
    assert loaded.use_cache is False
    assert loaded._description_texts == descriptions
    np.testing.assert_allclose(loaded.predict(TEST_UTTERANCES), before)

    # dump() must leave the live instance usable
    np.testing.assert_allclose(scorer.predict(TEST_UTTERANCES), before)


def test_registered_in_scoring_modules() -> None:
    from autointent.modules import SCORING_MODULES

    assert SCORING_MODULES["description_typesafe"] is TypeSafeDescriptionScorer


def test_create_clients_builds_real_sdk_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real SDK seam once (unmocked): `_create_clients` must build actual SDK client
    instances, not just satisfy the fake used everywhere else in this file. Requires the
    ``typesafe`` extra; skipped when it isn't installed. Makes no network call.
    """
    typesafe_sdk = pytest.importorskip("typesafe_sdk")
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-key-for-tests")

    sync_client, async_client = TypeSafeDescriptionScorer(model="jev-1.13.0", max_retries=2)._create_clients()

    assert isinstance(sync_client, typesafe_sdk.TypeSafeClient)
    assert isinstance(async_client, typesafe_sdk.AsyncTypeSafeClient)
