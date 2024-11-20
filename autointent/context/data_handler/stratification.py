from collections.abc import Sequence

import numpy as np
from datasets import Dataset as HFDataset
from numpy import typing as npt
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification

from .dataset import Dataset, Split


class StratifiedSplitter:
    def __init__(
        self, test_size: float, label_feature: str, random_seed: int, shuffle: bool = True,
    ) -> None:
        self.test_size = test_size
        self.label_feature = label_feature
        self.random_seed = random_seed
        self.shuffle = shuffle

    def __call__(self, dataset: HFDataset, multilabel: bool) -> tuple[Dataset, Dataset]:
        splits = self._split_multilabel(dataset) if multilabel else self._split(dataset)
        return dataset.select(splits[0]), dataset.select(splits[1])

    def _split(self, dataset: HFDataset) -> Sequence[npt.NDArray[np.int_]]:
        return train_test_split( # type: ignore[no-any-return]
            np.arange(len(dataset)),
            test_size=self.test_size,
            random_state=self.random_seed,
            shuffle=self.shuffle,
            stratify=dataset[self.label_feature],
        )

    def _split_multilabel(self, dataset: HFDataset) -> Sequence[npt.NDArray[np.int_]]:
        splitter = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[self.test_size, 1.0 - self.test_size],
        )
        return next(splitter.split(np.arange(len(dataset)), np.array(dataset[self.label_feature])))


def split_dataset(dataset: Dataset, random_seed: int) -> Dataset:
    splitter = StratifiedSplitter(
        test_size=0.25, label_feature=dataset.label_feature, random_seed=random_seed,
    )
    dataset[Split.TRAIN], dataset[Split.TEST] = splitter(dataset[Split.TRAIN], dataset.multilabel)
    return dataset
