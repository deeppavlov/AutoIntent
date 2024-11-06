from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt
from typing_extensions import Self

from autointent.context import Context
from autointent.context.optimization_info.data_models import Artifact
from autointent.custom_types import BaseMetadataDict
from autointent.metrics import METRIC_FN


class Module(ABC):
    metadata_dict_name: str = "metadata.json"
    metadata: BaseMetadataDict

    @abstractmethod
    def fit(self, *args: tuple[Any], **kwargs: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def score(self, context: Context, metric_fn: METRIC_FN) -> float:
        """
        calculates metric on test set and returns metric value
        """

    @abstractmethod
    def get_assets(self) -> Artifact:
        """
        return useful assets that represent intermediate data into context
        """

    @abstractmethod
    def clear_cache(self) -> None:
        """clear GPU/CPU memory"""

    @abstractmethod
    def dump(self, path: str) -> None:
        """dump all data needed for inference"""

    @abstractmethod
    def load(self, path: str) -> None:
        """load all data needed for inference"""

    @abstractmethod
    def predict(self, *args: list[str] | npt.NDArray[Any], **kwargs: dict[str, Any]) -> npt.NDArray[Any]:
        """inference"""

    def predict_with_metadata(
        self, *args: list[str] | npt.NDArray[Any], **kwargs: dict[str, Any],
    ) -> tuple[npt.NDArray[Any], list[dict[str, Any]] | None]:
        return self.predict(*args, **kwargs), None

    @classmethod
    @abstractmethod
    def from_context(cls, context: Context, **kwargs: dict[str, Any]) -> Self:
        pass
