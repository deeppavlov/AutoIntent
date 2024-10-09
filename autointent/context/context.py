from typing import Any

from chromadb import Collection

from .data_handler import DataHandler, Dataset
from .optimization_info import OptimizationInfo
from .vector_index import VectorIndex


class Context:
    def __init__(
        self,
        dataset: Dataset,
        test_dataset: Dataset | None,
        device: str,
        multilabel_generation_config: str,
        db_dir: str,
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
        self.vector_index = VectorIndex(db_dir, device, self.data_handler.multilabel, self.data_handler.n_classes)

        self.device = device
        self.multilabel = self.data_handler.multilabel
        self.n_classes = self.data_handler.n_classes
        self.seed = seed

    def get_best_collection(self) -> Collection:
        model_name = self.optimization_info.get_best_embedder()
        return self.vector_index.get_collection(model_name)

    def get_inference_config(self) -> dict[str, Any]:
        return {
            "metadata": {
                "device": self.device,
                "multilabel": self.multilabel,
                "n_classes": self.n_classes,
                "seed": self.seed,
                "db_dir": self.vector_index.db_dir,
            },
            "nodes_configs": self.optimization_info.get_best_trials(),
        }
