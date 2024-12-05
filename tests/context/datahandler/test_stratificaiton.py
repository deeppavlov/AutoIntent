from autointent.context.data_handler import Split
from autointent.context.data_handler._stratification import split_dataset


def test_train_test_split(dataset):
    dataset[Split.TRAIN], dataset[Split.TEST] = split_dataset(
        dataset,
        split=Split.TRAIN,
        test_size=0.2,
        random_seed=42,
    )

    assert Split.TRAIN in dataset
    assert Split.TEST in dataset
    assert dataset[Split.TRAIN].num_rows == 29
    assert dataset[Split.TEST].num_rows == 8
    assert dataset.get_n_classes(Split.TRAIN) == dataset.get_n_classes(Split.TEST)


def test_multilabel_train_test_split(dataset):
    dataset = dataset.to_multilabel().encode_labels()
    dataset[Split.TRAIN], dataset[Split.TEST] = split_dataset(
        dataset,
        split=Split.TRAIN,
        test_size=0.2,
        random_seed=42,
    )

    assert Split.TRAIN in dataset
    assert Split.TEST in dataset
    assert dataset[Split.TRAIN].num_rows == 30
    assert dataset[Split.TEST].num_rows == 7
    assert dataset.get_n_classes(Split.TRAIN) == dataset.get_n_classes(Split.TEST)
