from typing import Any

from autointent.custom_types import TASK_TYPES

from .data_handler import DataHandler
from .optimization_info import OptimizationInfo
from .vector_index import Index, VectorIndex


class Context:
    def __init__(
        self,
        multiclass_intent_records: list[dict[str, Any]],
        multilabel_utterance_records: list[dict[str, Any]],
        test_utterance_records: list[dict[str, Any]],
        device: str,
        mode: TASK_TYPES,
        multilabel_generation_config: str,
        regex_sampling: int,
        seed: int,
    ) -> None:
        self.data_handler = DataHandler(
            multiclass_intent_records,
            multilabel_utterance_records,
            test_utterance_records,
            mode,
            multilabel_generation_config,
            regex_sampling,
            seed,
        )
        self.optimization_info = OptimizationInfo()
        self.vector_index = VectorIndex(device, self.data_handler.multilabel, self.data_handler.n_classes)

        self.device = device
        self.multilabel = self.data_handler.multilabel
        self.n_classes = self.data_handler.n_classes
        self.seed = seed

    def get_best_index(self) -> Index:
        model_name = self.optimization_info.get_best_embedder()
        return self.vector_index.get_index(model_name)

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
