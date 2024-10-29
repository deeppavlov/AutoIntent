from dataclasses import asdict
from pathlib import Path
from typing import Any

from .data_handler import DataAugmenter, DataHandler, Dataset
from .optimization_info import OptimizationInfo
from .vector_index_client import VectorIndex, VectorIndexClient


class Context:
    def __init__(  # noqa: PLR0913
        self,
        dataset: Dataset,
        test_dataset: Dataset | None = None,
        device: str = "cpu",
        multilabel_generation_config: str | None = None,
        regex_sampling: int = 0,
        seed: int = 42,
        db_dir: str | Path | None = None,
        dump_dir: str | Path | None = None,
        force_multilabel: bool = False,
        embedder_batch_size: int = 1,
        embedder_max_length: int | None = None,
    ) -> None:
        augmenter = DataAugmenter(multilabel_generation_config, regex_sampling, seed)
        self.data_handler = DataHandler(
            dataset, test_dataset, random_seed=seed, force_multilabel=force_multilabel, augmenter=augmenter
        )
        self.optimization_info = OptimizationInfo()
        self.db_dir: Path | None = Path(db_dir) if db_dir is not None else None
        self.vector_index_client = VectorIndexClient(device, str(self.db_dir), embedder_batch_size, embedder_max_length)

        self.embedder_batch_size = embedder_batch_size
        self.device = device
        self.multilabel = self.data_handler.multilabel
        self.n_classes = self.data_handler.n_classes
        self.seed = seed
        self.dump_dir = Path.cwd() / "modules_dumps" if dump_dir is None else Path(dump_dir)

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
