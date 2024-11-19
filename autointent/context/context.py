import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from autointent.configs.optimization_cli import (
    DataConfig,
    EmbedderConfig,
    LoggingConfig,
    VectorIndexConfig,
)

from .data_handler import DataHandler, Dataset
from .optimization_info import OptimizationInfo
from .utils import NumpyEncoder, load_data
from .vector_index_client import VectorIndex, VectorIndexClient


class Context:
    data_handler: DataHandler
    vector_index_client: VectorIndexClient
    optimization_info: OptimizationInfo

    def __init__(
        self,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self._logger = logging.getLogger(__name__)

    def configure_logging(self, config: LoggingConfig) -> None:
        self.logging_config = config
        self.optimization_info = OptimizationInfo()

    def configure_vector_index(self, config: VectorIndexConfig, embedder_config: EmbedderConfig | None = None) -> None:
        self.vector_index_config = config
        if embedder_config is None:
            embedder_config = EmbedderConfig()
        self.embedder_config = embedder_config

        self.vector_index_client = VectorIndexClient(
            self.vector_index_config.device,
            self.vector_index_config.db_dir,
            self.embedder_config.batch_size,
            self.embedder_config.max_length,
        )

    def configure_data(self, config: DataConfig) -> None:
        self.data_handler = DataHandler(
            dataset=load_data(config.train_path),
            test_dataset=None if config.test_path is None else load_data(config.test_path),
            random_seed=self.seed,
            force_multilabel=config.force_multilabel,
        )

    def set_datasets(
        self, train_data: Dataset, val_data: Dataset | None = None, force_multilabel: bool = False
    ) -> None:
        self.data_handler = DataHandler(
            dataset=train_data, test_dataset=val_data, random_seed=self.seed, force_multilabel=force_multilabel
        )

    def get_best_index(self) -> VectorIndex:
        model_name = self.optimization_info.get_best_embedder()
        return self.vector_index_client.get_index(model_name)

    def get_inference_config(self) -> dict[str, Any]:
        nodes_configs = [asdict(cfg) for cfg in self.optimization_info.get_inference_nodes_config()]
        for cfg in nodes_configs:
            cfg.pop("_target_")
        return {
            "metadata": {
                "device": self.get_device(),
                "multilabel": self.is_multilabel(),
                "n_classes": self.get_n_classes(),
                "seed": self.seed,
            },
            "nodes_configs": nodes_configs,
        }

    def dump(self) -> None:
        self._logger.debug("dumping logs...")
        optimization_results = self.optimization_info.dump_evaluation_results()

        logs_dir = self.logging_config.dirpath
        if logs_dir is None:
            msg = "something's wrong with LoggingConfig"
            raise ValueError(msg)

        # create appropriate directory
        logs_dir.mkdir(parents=True, exist_ok=True)

        # dump search space and evaluation results
        logs_path = logs_dir / "logs.json"
        with logs_path.open("w") as file:
            json.dump(optimization_results, file, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        # config_path = logs_dir / "config.yaml"
        # with config_path.open("w") as file:
        #     yaml.dump(self.config, file)

        # self._logger.info(make_report(optimization_results, nodes=nodes))

        # dump train and test data splits
        train_data, test_data = self.data_handler.dump()
        train_path = logs_dir / "train_data.json"
        test_path = logs_dir / "test_data.json"
        with train_path.open("w") as file:
            json.dump(train_data, file, indent=4, ensure_ascii=False)
        with test_path.open("w") as file:
            json.dump(test_data, file, indent=4, ensure_ascii=False)

        self._logger.info("logs and other assets are saved to %s", logs_dir)

        # dump optimization results (config for inference)
        inference_config = self.get_inference_config()
        inference_config_path = logs_dir / "inference_config.yaml"
        with inference_config_path.open("w") as file:
            yaml.dump(inference_config, file)

    def get_db_dir(self) -> Path:
        return self.vector_index_client.db_dir

    def get_device(self) -> str:
        return self.vector_index_client.device

    def get_batch_size(self) -> int:
        return self.vector_index_client.embedder_batch_size

    def get_max_length(self) -> int | None:
        return self.vector_index_client.embedder_max_length

    def get_dump_dir(self) -> Path | None:
        if self.logging_config.dump_modules:
            return self.logging_config.dump_dir
        return None

    def is_multilabel(self) -> bool:
        return self.data_handler.multilabel

    def get_n_classes(self) -> int:
        return self.data_handler.n_classes

    def is_ram_to_clear(self) -> bool:
        return self.logging_config.clear_ram

    def has_saved_modules(self) -> bool:
        node_types = ["regexp", "retrieval", "scoring", "prediction"]
        return any(len(self.optimization_info.modules.get(nt)) > 0 for nt in node_types)
