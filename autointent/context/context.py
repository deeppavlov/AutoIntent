from dataclasses import asdict
from typing import Any

from autointent.custom_types import TASK_TYPES

from .data_handler import DataHandler
from .optimization_info import OptimizationInfo
from .vector_index_client import VectorIndex, VectorIndexClient


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
        db_dir: str,
        dump_dir: str,
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
        self.vector_index_client = VectorIndexClient(
            device, db_dir
        )

        self.device = device
        self.multilabel = self.data_handler.multilabel
        self.n_classes = self.data_handler.n_classes
        self.seed = seed
        self.db_dir = db_dir
        self.dump_dir = dump_dir

    def get_best_index(self) -> VectorIndex:
        model_name = self.optimization_info.get_best_embedder()
        return self.vector_index_client.get_index(model_name)

    def get_inference_config(self) -> dict[str, Any]:
        nodes_configs = [asdict(cfg) for cfg in self.optimization_info.get_inference_nodes_config()]
        for cfg in nodes_configs:
            cfg.pop("_target_")
        return {
            "metadata": {
                "device": self.device,
                "multilabel": self.multilabel,
                "n_classes": self.n_classes,
                "seed": self.seed,
            },
            "nodes_configs": nodes_configs,
        }
