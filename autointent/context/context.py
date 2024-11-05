import importlib.resources as ires
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import ListConfig

from autointent.configs.optimization_cli import (
    AugmentationConfig,
    DataConfig,
    EmbedderConfig,
    LoggingConfig,
    VectorIndexConfig,
)

from .data_handler import DataAugmenter, DataHandler, Dataset
from .optimization_info import OptimizationInfo
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

    def config_logs(self, config: LoggingConfig) -> None:
        self.logging_config = config
        self.optimization_info = OptimizationInfo()

    def config_vector_index(self, config: VectorIndexConfig, embedder_config: EmbedderConfig | None = None) -> None:
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

    def config_data(self, config: DataConfig, augmentation_config: AugmentationConfig | None = None) -> None:
        if augmentation_config is not None:
            self.augmentation_config = AugmentationConfig()
            augmenter = DataAugmenter(
                self.augmentation_config.multilabel_generation_config,
                self.augmentation_config.regex_sampling,
                self.seed,
            )
        else:
            augmenter = None

        self.data_handler = DataHandler(
            dataset=load_data(config.train_path),
            test_dataset=None if config.test_path is None else load_data(config.test_path),
            random_seed=self.seed,
            force_multilabel=config.force_multilabel,
            augmenter=augmenter,
        )

        self.multilabel = self.data_handler.multilabel
        self.n_classes = self.data_handler.n_classes

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

    def dump(self) -> None:
        self._logger.debug("dumping logs...")
        optimization_results = self.optimization_info.dump_evaluation_results()

        logs_dir = self.logging_config.dirpath

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

    def get_dump_dir(self) -> Path:
        return self.logging_config.dump_dir

class NumpyEncoder(json.JSONEncoder):
    """Helper for dumping logs. Problem explained: https://stackoverflow.com/q/50916422"""

    def default(self, obj: Any) -> str | int | float | list[Any] | Any:  # noqa: ANN401
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, ListConfig):
            return list(obj)
        return super().default(obj)


def load_data(data_path: str | Path) -> Dataset:
    """load data from the given path or load sample data which is distributed along with the autointent package"""
    if data_path == "default-multiclass":
        with ires.files("autointent.datafiles").joinpath("banking77.json").open() as file:
            res = json.load(file)
    elif data_path == "default-multilabel":
        with ires.files("autointent.datafiles").joinpath("dstc3-20shot.json").open() as file:
            res = json.load(file)
    else:
        with Path(data_path).open() as file:
            res = json.load(file)

    return Dataset.model_validate(res)
