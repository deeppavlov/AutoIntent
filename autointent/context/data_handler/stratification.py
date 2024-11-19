import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification

from .dataset import Dataset, Split


class StratifiedSplitter:
    def __init__(self, test_size: float, random_seed: int) -> None:
        self.test_size = test_size
        self.random_seed = random_seed

    def __call__(self, dataset: Dataset, multilabel: bool) -> tuple[Dataset, Dataset]:
        splits = self._split_multilabel(dataset) if multilabel else self._split(dataset)
        return dataset.select(splits[0]), dataset.select(splits[1])

    def _split(self, dataset: Dataset) -> ...:
        return train_test_split(
            np.arange(len(dataset)),
            test_size=self.test_size,
            random_state=self.random_seed,
            shuffle=True,
            stratify=dataset[dataset.label_column],
        )

    def _split_multilabel(self, dataset: Dataset) -> ...:
        splitter = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[self.test_size, 1.0 - self.test_size],
        )
        return next(splitter.split(np.arange(len(dataset)), np.array(dataset[dataset.label_column])))


def split(dataset: Dataset, random_seed: int) -> Dataset:
    splitter = StratifiedSplitter(test_size=0.25, random_seed=random_seed)
    dataset[Split.TRAIN], dataset[Split.TEST] = splitter(dataset[Split.TRAIN], dataset.multilabel)
    return dataset
