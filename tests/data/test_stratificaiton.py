import pytest

from autointent.context.data_handler._stratification import split_dataset
from autointent.custom_types import Split


def test_train_test_split(dataset_unsplitted):
    dataset = dataset_unsplitted
    kwargs = {
        "dataset": dataset,
        "split": Split.TRAIN,
        "test_size": 0.5,
        "random_seed": 42,
    }
    with pytest.raises(ValueError):  # noqa: PT011
        split_dataset(**kwargs)

    kwargs["allow_oos_in_train"] = False
    dataset[Split.TRAIN], dataset[Split.TEST] = split_dataset(**kwargs)

    assert Split.TRAIN in dataset
    assert Split.TEST in dataset
    assert dataset[Split.TRAIN].num_rows == 18
    assert dataset[Split.TEST].num_rows == 18
    assert dataset.get_n_classes(Split.TRAIN) == dataset.get_n_classes(Split.TEST)


def test_multilabel_train_test_split(dataset_unsplitted):
    dataset = dataset_unsplitted
    dataset = dataset.to_multilabel()
    dataset[Split.TRAIN], dataset[Split.TEST] = split_dataset(
        dataset,
        split=Split.TRAIN,
        test_size=0.5,
        random_seed=42,
        allow_oos_in_train=False,
    )

    assert Split.TRAIN in dataset
    assert Split.TEST in dataset
    assert dataset[Split.TRAIN].num_rows == 18
    assert dataset[Split.TEST].num_rows == 18
    assert dataset.get_n_classes(Split.TRAIN) == dataset.get_n_classes(Split.TEST)


def test_multilabel_train_test_split_few_shot(dataset_unsplitted):
    dataset = dataset_unsplitted
    dataset = dataset.to_multilabel()
    dataset[Split.TRAIN], dataset[Split.TEST] = split_dataset(
        dataset,
        split=Split.TRAIN,
        test_size=0.5,
        random_seed=42,
        allow_oos_in_train=False,
        is_few_shot=True,
        examples_per_intent=2,
    )

    assert Split.TRAIN in dataset
    assert Split.TEST in dataset
    assert dataset[Split.TRAIN].num_rows == 8
    assert dataset[Split.TEST].num_rows == 28
    assert dataset.get_n_classes(Split.TRAIN) == dataset.get_n_classes(Split.TEST)


def test_multiclass_train_test_split_few_shot(dataset_unsplitted):
    dataset = dataset_unsplitted
    dataset[Split.TRAIN], dataset[Split.TEST] = split_dataset(
        dataset,
        split=Split.TRAIN,
        test_size=0.5,
        random_seed=42,
        allow_oos_in_train=False,
        is_few_shot=True,
        examples_per_intent=2,
    )

    assert Split.TRAIN in dataset
    assert Split.TEST in dataset
    assert dataset[Split.TRAIN].num_rows == 8
    assert dataset[Split.TEST].num_rows == 28
    assert dataset.get_n_classes(Split.TRAIN) == dataset.get_n_classes(Split.TEST)
