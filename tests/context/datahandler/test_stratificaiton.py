from autointent.context.data_handler._dataset import Split
from autointent.context.data_handler._stratification import split_dataset


def test_train_test_split(dataset):
    dataset = split_dataset(dataset, random_seed=42)

    assert Split.TRAIN in dataset
    assert Split.TEST in dataset
    assert dataset[Split.TRAIN].num_rows == 11
    assert dataset[Split.TEST].num_rows == 4
    assert dataset.get_n_classes(Split.TRAIN) == dataset.get_n_classes(Split.TEST)


def test_multilabel_train_test_split(dataset):
    dataset = dataset.to_multilabel().encode_labels()
    dataset = split_dataset(dataset, random_seed=42)

    assert Split.TRAIN in dataset
    assert Split.TEST in dataset
    assert dataset[Split.TRAIN].num_rows == 12
    assert dataset[Split.TEST].num_rows == 3
    assert dataset.get_n_classes(Split.TRAIN) == dataset.get_n_classes(Split.TEST)
