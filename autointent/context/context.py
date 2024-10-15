from typing import Any

from .data_handler import DataHandler, Dataset
from .optimization_info import OptimizationInfo
from .vector_index_client import VectorIndex, VectorIndexClient


class Context:
    def __init__(
        self,
        dataset: Dataset,
        test_dataset: Dataset | None,
        device: str,
        multilabel_generation_config: str,
        regex_sampling: int,
        seed: int,
    ) -> None:
        self.data_handler = DataHandler(
            dataset,
            test_dataset,
            multilabel_generation_config,
            regex_sampling,
            seed,
        )
        self.optimization_info = OptimizationInfo()
        self.vector_index_client = VectorIndexClient(device, self.data_handler.multilabel, self.data_handler.n_classes)

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
