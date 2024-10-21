from typing import Any

from .data_handler import DataHandler, Dataset
from .optimization_info import OptimizationInfo
from .vector_index_client import VectorIndex, VectorIndexClient


class Context:
    def __init__(
        self,
        dataset: Dataset,
        test_dataset: Dataset | None = None,
        device: str | None = None,
        multilabel_generation_config: str | None = None,
        regex_sampling: int = 0,
        seed: int = 42,
    ) -> None:
        self.data_handler = DataHandler(
            dataset,
            test_dataset,
            multilabel_generation_config,
            regex_sampling,
            random_seed=seed,
        )
        self.optimization_info = OptimizationInfo()
        self.vector_index_client = VectorIndexClient(
            self.data_handler.multilabel, self.data_handler.n_classes, device=device
        )

        self.device = device
        self.multilabel = self.data_handler.multilabel
        self.n_classes = self.data_handler.n_classes
        self.seed = seed

    def get_best_index(self) -> VectorIndex:
        model_name = self.optimization_info.get_best_embedder()
        return self.vector_index_client.get_index(model_name)

    def get_inference_config(self) -> dict[str, Any]:
        return {
            "metadata": {
                "device": self.device,
                "multilabel": self.multilabel,
                "n_classes": self.n_classes,
                "seed": self.seed,
            },
            "nodes_configs": self.optimization_info.get_best_trials(),
        }
