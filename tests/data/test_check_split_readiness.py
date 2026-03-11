"""Unit tests for check_split_readiness and SplitReadinessResult."""

import pytest

from autointent import Dataset
from autointent.context.data_handler import (
    SplitReadinessResult,
    check_split_readiness,
)
from autointent.custom_types import Split


@pytest.fixture
def dataset_enough_samples():
    """Multiclass dataset with ≥2 samples per class (no OOS). Ready for stratification."""
    return Dataset.from_dict(
        {
            "train": [
                {"utterance": "a1", "label": 0},
                {"utterance": "a2", "label": 0},
                {"utterance": "b1", "label": 1},
                {"utterance": "b2", "label": 1},
                {"utterance": "c1", "label": 2},
                {"utterance": "c2", "label": 2},
            ],
            "test": [
                {"utterance": "t1", "label": 0},
                {"utterance": "t2", "label": 1},
                {"utterance": "t3", "label": 2},
            ],
            "intents": [
                {"id": 0, "regex_full_match": [], "regex_partial_match": []},
                {"id": 1, "regex_full_match": [], "regex_partial_match": []},
                {"id": 2, "regex_full_match": [], "regex_partial_match": []},
            ],
        }
    )


@pytest.fixture
def dataset_underpopulated():
    """Multiclass dataset with one class having only 1 sample. Not ready for stratification."""
    return Dataset.from_dict(
        {
            "train": [
                {"utterance": "a1", "label": 0},
                {"utterance": "a2", "label": 0},
                {"utterance": "b1", "label": 1},
            ],
            "test": [
                {"utterance": "t1", "label": 0},
                {"utterance": "t2", "label": 1},
            ],
            "intents": [
                {"id": 0, "regex_full_match": [], "regex_partial_match": []},
                {"id": 1, "regex_full_match": [], "regex_partial_match": []},
            ],
        }
    )


@pytest.fixture
def dataset_two_classes_barely_enough():
    """Two classes with exactly 2 samples each. Ready for default min_samples_per_class=2."""
    return Dataset.from_dict(
        {
            "train": [
                {"utterance": "a1", "label": 0},
                {"utterance": "a2", "label": 0},
                {"utterance": "b1", "label": 1},
                {"utterance": "b2", "label": 1},
            ],
            "test": [
                {"utterance": "t1", "label": 0},
                {"utterance": "t2", "label": 1},
            ],
            "intents": [
                {"id": 0, "regex_full_match": [], "regex_partial_match": []},
                {"id": 1, "regex_full_match": [], "regex_partial_match": []},
            ],
        }
    )


def test_check_split_readiness_ready_when_enough_samples(dataset_enough_samples):
    """When every class has ≥ min_samples_per_class, result is ready."""
    result = check_split_readiness(
        dataset_enough_samples,
        split=Split.TRAIN,
        test_size=0.3,
        allow_oos_in_train=False,
    )
    assert isinstance(result, SplitReadinessResult)
    assert result.ready is True
    assert result.underpopulated_classes == []
    assert result.min_samples_per_class_required == 2
    assert result.reason is None


def test_check_split_readiness_not_ready_underpopulated(dataset_underpopulated):
    """When at least one class has fewer than min samples, result is not ready."""
    result = check_split_readiness(
        dataset_underpopulated,
        split=Split.TRAIN,
        test_size=0.3,
        allow_oos_in_train=False,
    )
    assert result.ready is False
    assert result.min_samples_per_class_required == 2
    assert len(result.underpopulated_classes) == 1
    label, count = result.underpopulated_classes[0]
    assert label == 1
    assert count == 1
    assert result.reason is not None
    assert "class 1" in result.reason or "1" in result.reason
    assert "1 (need 2)" in result.reason


def test_check_split_readiness_missing_split(dataset_enough_samples):
    """When split is not in dataset, result is not ready with reason."""
    result = check_split_readiness(
        dataset_enough_samples,
        split="nonexistent_split",
        test_size=0.3,
    )
    assert result.ready is False
    assert result.underpopulated_classes == []
    assert "nonexistent_split" in result.reason


def test_check_split_readiness_oos_allow_none(dataset_unsplitted):
    """When dataset has OOS and allow_oos_in_train is None, result is not ready."""
    result = check_split_readiness(
        dataset_unsplitted,
        split=Split.TRAIN,
        test_size=0.5,
        allow_oos_in_train=None,
    )
    assert result.ready is False
    assert "OOS" in result.reason or "allow_oos_in_train" in result.reason


def test_check_split_readiness_oos_allow_false_enough_in_domain(dataset_unsplitted):
    """With OOS and allow_oos_in_train=False, in-domain classes are checked; clinc subset has enough."""
    result = check_split_readiness(
        dataset_unsplitted,
        split=Split.TRAIN,
        test_size=0.5,
        allow_oos_in_train=False,
    )
    assert result.ready is True
    assert result.underpopulated_classes == []
    assert result.reason is None


def test_check_split_readiness_min_samples_per_class_param(dataset_two_classes_barely_enough):
    """Custom min_samples_per_class is respected."""
    result = check_split_readiness(
        dataset_two_classes_barely_enough,
        split=Split.TRAIN,
        test_size=0.3,
        min_samples_per_class=2,
        allow_oos_in_train=False,
    )
    assert result.ready is True

    result_strict = check_split_readiness(
        dataset_two_classes_barely_enough,
        split=Split.TRAIN,
        test_size=0.3,
        min_samples_per_class=3,
        allow_oos_in_train=False,
    )
    assert result_strict.ready is False
    assert len(result_strict.underpopulated_classes) == 2
    assert result_strict.min_samples_per_class_required == 3


def test_check_split_readiness_multilabel_returns_ready(dataset_unsplitted):
    """Multilabel datasets return ready=True (multilabel stratification is not validated)."""
    dataset = dataset_unsplitted.to_multilabel()
    result = check_split_readiness(
        dataset,
        split=Split.TRAIN,
        test_size=0.5,
        allow_oos_in_train=False,
    )
    assert result.ready is True
    assert result.underpopulated_classes == []


def test_check_split_readiness_consistent_with_split_dataset(dataset_enough_samples):
    """When check_split_readiness says ready, split_dataset does not raise."""
    result = check_split_readiness(
        dataset_enough_samples,
        split=Split.TRAIN,
        test_size=0.5,
        allow_oos_in_train=False,
    )
    assert result.ready is True
    from autointent.context.data_handler import split_dataset

    train, test = split_dataset(
        dataset_enough_samples,
        split=Split.TRAIN,
        test_size=0.5,
        random_seed=42,
        allow_oos_in_train=False,
    )
    assert len(train) > 0
    assert len(test) > 0


def test_check_split_readiness_underpopulated_implies_split_raises(dataset_underpopulated):
    """When check_split_readiness says not ready (underpopulated), split_dataset raises."""
    result = check_split_readiness(
        dataset_underpopulated,
        split=Split.TRAIN,
        test_size=0.3,
        allow_oos_in_train=False,
    )
    assert result.ready is False
    from autointent.context.data_handler import split_dataset

    with pytest.raises(ValueError, match=r"least populated|too few"):
        split_dataset(
            dataset_underpopulated,
            split=Split.TRAIN,
            test_size=0.3,
            random_seed=42,
            allow_oos_in_train=False,
        )
