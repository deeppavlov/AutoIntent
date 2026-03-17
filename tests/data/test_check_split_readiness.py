"""Unit tests for check_split_readiness and SplitReadinessResult."""

import pytest

from autointent import Dataset
from autointent.configs import DataConfig
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
                {"utterance": "a3", "label": 0},
                {"utterance": "b1", "label": 1},
                {"utterance": "b2", "label": 1},
                {"utterance": "b3", "label": 1},
                {"utterance": "c1", "label": 2},
                {"utterance": "c2", "label": 2},
                {"utterance": "c3", "label": 2},
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
def dataset_three_classes_two_each():
    """3 classes, 2 samples each (no OOS). Useful for split-size feasibility tests."""
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
        config=DataConfig(validation_size=0.3, separation_ratio=None),
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
        config=DataConfig(validation_size=0.3, separation_ratio=None),
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
        config=DataConfig(validation_size=0.3, separation_ratio=None),
    )
    assert result.ready is False
    assert result.underpopulated_classes == []
    assert "nonexistent_split" in result.reason


def test_check_split_readiness_oos_allow_none(dataset_unsplitted):
    """When dataset has OOS and allow_oos_in_train is None, result is not ready."""
    with pytest.raises(ValueError, match="allow_oos_in_train"):
        check_split_readiness(
            dataset_unsplitted,
            split=Split.TRAIN,
            config=DataConfig(validation_size=0.5, separation_ratio=None),
            allow_oos_in_train=None,
        )


def test_check_split_readiness_oos_allow_false_enough_in_domain(dataset_unsplitted):
    """With OOS and allow_oos_in_train=False, in-domain classes are checked; clinc subset has enough."""
    result = check_split_readiness(
        dataset_unsplitted,
        split=Split.TRAIN,
        config=DataConfig(validation_size=0.5, separation_ratio=None),
        allow_oos_in_train=False,
    )
    assert result.ready is True
    assert result.underpopulated_classes == []
    assert result.reason is None


def test_check_split_readiness_multiclass_too_small_test_split(dataset_three_classes_two_each):
    """Even with >=2/class, stratification can fail if test split can't include all classes."""
    result = check_split_readiness(
        dataset_three_classes_two_each,
        split=Split.TRAIN,
        config=DataConfig(validation_size=0.1, separation_ratio=None),
        allow_oos_in_train=False,
    )
    assert result.ready is False
    assert result.underpopulated_classes == []
    assert result.reason is not None
    assert "too few test samples" in result.reason


def test_check_split_readiness_multiclass_too_small_train_split(dataset_three_classes_two_each):
    """Even with >=2/class, stratification can fail if train split can't include all classes."""
    result = check_split_readiness(
        dataset_three_classes_two_each,
        split=Split.TRAIN,
        config=DataConfig(validation_size=0.8, separation_ratio=None),
        allow_oos_in_train=False,
    )
    assert result.ready is False
    assert result.underpopulated_classes == []
    assert result.reason is not None
    assert "too few train samples" in result.reason


def test_check_split_readiness_min_samples_per_class_param(dataset_two_classes_barely_enough):
    """Custom min_samples_per_class is respected."""
    result = check_split_readiness(
        dataset_two_classes_barely_enough,
        split=Split.TRAIN,
        config=DataConfig(validation_size=0.3, separation_ratio=None),
        allow_oos_in_train=False,
    )
    assert result.ready is True

    result_strict = check_split_readiness(
        dataset_two_classes_barely_enough,
        split=Split.TRAIN,
        config=DataConfig(scheme="cv", n_folds=3, separation_ratio=None),
        allow_oos_in_train=False,
    )
    assert result_strict.ready is False
    assert len(result_strict.underpopulated_classes) == 2
    assert result_strict.min_samples_per_class_required == 3


def test_check_split_readiness_multilabel_returns_ready():
    """Multilabel datasets are checked by per-label positive counts."""
    dataset = Dataset.from_dict(
        {
            "train": [
                {"utterance": "x1", "label": [1, 0, 1]},
                {"utterance": "x2", "label": [1, 0, 0]},
                {"utterance": "x3", "label": [0, 1, 0]},
                {"utterance": "x4", "label": [0, 0, 1]},
            ],
            "intents": [
                {"id": 0, "regex_full_match": [], "regex_partial_match": []},
                {"id": 1, "regex_full_match": [], "regex_partial_match": []},
                {"id": 2, "regex_full_match": [], "regex_partial_match": []},
            ],
        }
    )

    # label 1 appears only once -> not ready for min_samples_per_class=2
    result = check_split_readiness(
        dataset,
        split=Split.TRAIN,
        config=DataConfig(validation_size=0.5, separation_ratio=None),
        allow_oos_in_train=False,
    )
    assert result.ready is False
    assert result.underpopulated_classes == [(1, 1)]
    assert result.reason is not None


def test_check_split_readiness_marks_declared_but_unseen_intent_as_underpopulated():
    """Intents with 0 samples should be flagged so callers can filter them out."""
    dataset = Dataset.from_dict(
        {
            "train": [
                {"utterance": "a1", "label": 0},
                {"utterance": "a2", "label": 0},
                {"utterance": "b1", "label": 1},
                {"utterance": "b2", "label": 1},
            ],
            # Declare 3 intents, but only provide samples for ids 0 and 1.
            "intents": [
                {"id": 0, "regex_full_match": [], "regex_partial_match": []},
                {"id": 1, "regex_full_match": [], "regex_partial_match": []},
                {"id": 2, "regex_full_match": [], "regex_partial_match": []},
            ],
        }
    )

    result = check_split_readiness(
        dataset,
        split=Split.TRAIN,
        config=DataConfig(validation_size=0.5, separation_ratio=None),
        allow_oos_in_train=False,
    )
    assert result.ready is False
    assert (2, 0) in result.underpopulated_classes
    assert result.reason is not None


def test_check_split_readiness_multilabel_oos_allow_true_checks_oos_label():
    """Multilabel + OOS + allow_oos_in_train=True should not crash and should include OOS label."""
    dataset = Dataset.from_dict(
        {
            "train": [
                {"utterance": "x1", "label": [1, 0]},
                {"utterance": "x2", "label": [1, 0]},
                {"utterance": "x3", "label": [0, 1]},
                {"utterance": "x4", "label": [0, 1]},
                {"utterance": "oos1", "label": None},
            ],
            "intents": [
                {"id": 0, "regex_full_match": [], "regex_partial_match": []},
                {"id": 1, "regex_full_match": [], "regex_partial_match": []},
            ],
        }
    )

    result = check_split_readiness(
        dataset,
        split=Split.TRAIN,
        config=DataConfig(validation_size=0.5, separation_ratio=None),
        allow_oos_in_train=True,
    )
    assert result.ready is False
    # OOS indicator label is appended -> index == n_classes == 2
    assert (2, 1) in result.underpopulated_classes
    assert result.reason is not None


def test_check_split_readiness_multilabel_oos_allow_true_ready_when_oos_sufficient():
    """When OOS count meets minimum, multilabel readiness can be true."""
    dataset = Dataset.from_dict(
        {
            "train": [
                {"utterance": "x1", "label": [1, 0]},
                {"utterance": "x2", "label": [1, 0]},
                {"utterance": "x3", "label": [0, 1]},
                {"utterance": "x4", "label": [0, 1]},
                {"utterance": "oos1", "label": None},
                {"utterance": "oos2", "label": None},
            ],
            "intents": [
                {"id": 0, "regex_full_match": [], "regex_partial_match": []},
                {"id": 1, "regex_full_match": [], "regex_partial_match": []},
            ],
        }
    )

    result = check_split_readiness(
        dataset,
        split=Split.TRAIN,
        config=DataConfig(validation_size=0.5, separation_ratio=None),
        allow_oos_in_train=True,
    )
    assert result.ready is True
    assert result.underpopulated_classes == []
    assert result.reason is None


def test_split_dataset_multilabel_oos_allow_true_does_not_raise():
    """Sanity-check: split_dataset supports multilabel+OOS when allow_oos_in_train=True."""
    dataset = Dataset.from_dict(
        {
            "train": [
                {"utterance": "x1", "label": [1, 0]},
                {"utterance": "x2", "label": [1, 0]},
                {"utterance": "x3", "label": [0, 1]},
                {"utterance": "x4", "label": [0, 1]},
                {"utterance": "oos1", "label": None},
                {"utterance": "oos2", "label": None},
            ],
            "intents": [
                {"id": 0, "regex_full_match": [], "regex_partial_match": []},
                {"id": 1, "regex_full_match": [], "regex_partial_match": []},
            ],
        }
    )
    from autointent.context.data_handler import split_dataset

    train, test = split_dataset(
        dataset,
        split=Split.TRAIN,
        test_size=0.5,
        random_seed=42,
        allow_oos_in_train=True,
    )
    assert len(train) > 0
    assert len(test) > 0


def test_check_split_readiness_consistent_with_split_dataset(dataset_enough_samples):
    """When check_split_readiness says ready, split_dataset does not raise."""
    result = check_split_readiness(
        dataset_enough_samples,
        split=Split.TRAIN,
        config=DataConfig(validation_size=0.5, separation_ratio=None),
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
        config=DataConfig(validation_size=0.3),
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


def test_stratified_splitter_multilabel_allow_oos_all_oos_raises_value_error():
    """Multilabel OOS mapping needs an in-domain row to infer label dimensionality."""
    from datasets import Dataset as HFDataset

    from autointent.context.data_handler._stratification import StratifiedSplitter

    hf_ds = HFDataset.from_list(
        [
            {"utterance": "oos1", "label": None},
            {"utterance": "oos2", "label": None},
        ]
    )
    splitter = StratifiedSplitter(test_size=0.5, label_feature="label", random_seed=0)
    with pytest.raises(ValueError, match=r"only OOS|infer multilabel dimensionality"):
        splitter.get_stratify_inputs(hf_ds, multilabel=True, allow_oos_in_train=True)
