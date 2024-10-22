from dataclasses import asdict
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
        db_dir: str,
        dump_dir: str,
        force_multilabel: bool = False,
        embedder_batch_size: int = 1,
    ) -> None:
        self.data_handler = DataHandler(
            dataset,
            test_dataset,
            multilabel_generation_config,
            regex_sampling,
            random_seed=seed,
            force_multilabel=force_multilabel,
        )
        self.optimization_info = OptimizationInfo()
        self.vector_index_client = VectorIndexClient(device, db_dir, embedder_batch_size)

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
