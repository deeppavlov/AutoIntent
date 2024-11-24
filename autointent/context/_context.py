"""Context manager for configuring and managing data handling, vector indexing, and optimization."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from autointent.configs import (
    AugmentationConfig,
    DataConfig,
    EmbedderConfig,
    LoggingConfig,
    VectorIndexConfig,
)

from ._utils import NumpyEncoder, load_data
from .data_handler import DataAugmenter, DataHandler, Dataset
from .optimization_info import OptimizationInfo
from .vector_index_client import VectorIndex, VectorIndexClient


class Context:
    """
    Context manager for configuring and managing data handling, vector indexing, and optimization.

    This class provides methods to set up logging, configure data and vector index components,
    manage datasets, and retrieve various configurations for inference and optimization.
    """

    data_handler: DataHandler
    vector_index_client: VectorIndexClient
    optimization_info: OptimizationInfo

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize the Context object with a specified random seed.

        :param seed: Random seed for reproducibility, defaults to 42.
        """
        self.seed = seed
        self._logger = logging.getLogger(__name__)

    def configure_logging(self, config: LoggingConfig) -> None:
        """
        Configure logging settings.

        :param config: Logging configuration settings.
        """
        self.logging_config = config
        self.optimization_info = OptimizationInfo()

    def configure_vector_index(self, config: VectorIndexConfig, embedder_config: EmbedderConfig | None = None) -> None:
        """
        Configure the vector index client and embedder.

        :param config: Configuration for the vector index.
        :param embedder_config: Configuration for the embedder. If None, a default EmbedderConfig is used.
        """
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

    def configure_data(self, config: DataConfig, augmentation_config: AugmentationConfig | None = None) -> None:
        """
        Configure data handling and augmentation.

        :param config: Configuration for the data handling process.
        :param augmentation_config: Configuration for data augmentation. If None, no augmentation is applied.
        """
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

    def set_datasets(
        self, train_data: Dataset, val_data: Dataset | None = None, force_multilabel: bool = False
    ) -> None:
        """
        Set the datasets for training and validation.

        :param train_data: Training dataset.
        :param val_data: Validation dataset. If None, only training data is used.
        :param force_multilabel: Whether to force multilabel classification.
        """
        self.data_handler = DataHandler(
            dataset=train_data, test_dataset=val_data, random_seed=self.seed, force_multilabel=force_multilabel
        )

    def get_best_index(self) -> VectorIndex:
        """
        Retrieve the best vector index based on optimization results.

        :return: Best vector index object.
        """
        model_name = self.optimization_info.get_best_embedder()
        return self.vector_index_client.get_index(model_name)

    def get_inference_config(self) -> dict[str, Any]:
        """
        Generate configuration settings for inference.

        :return: Dictionary containing inference configuration.
        """
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
        """
        Save logs, configurations, and datasets to disk.

        Dumps evaluation results, training/test data splits, and inference configurations
        to the specified logging directory.
        """
        self._logger.debug("dumping logs...")
        optimization_results = self.optimization_info.dump_evaluation_results()

        logs_dir = self.logging_config.dirpath
        if logs_dir is None:
            msg = "something's wrong with LoggingConfig"
            raise ValueError(msg)

        logs_dir.mkdir(parents=True, exist_ok=True)

        logs_path = logs_dir / "logs.json"
        with logs_path.open("w") as file:
            json.dump(optimization_results, file, indent=4, ensure_ascii=False, cls=NumpyEncoder)

        train_data, test_data = self.data_handler.dump()
        train_path = logs_dir / "train_data.json"
        test_path = logs_dir / "test_data.json"
        with train_path.open("w") as file:
            json.dump(train_data, file, indent=4, ensure_ascii=False)
        with test_path.open("w") as file:
            json.dump(test_data, file, indent=4, ensure_ascii=False)

        self._logger.info("logs and other assets are saved to %s", logs_dir)

        inference_config = self.get_inference_config()
        inference_config_path = logs_dir / "inference_config.yaml"
        with inference_config_path.open("w") as file:
            yaml.dump(inference_config, file)

    def get_db_dir(self) -> Path:
        """
        Get the database directory of the vector index.

        :return: Path to the database directory.
        """
        return self.vector_index_client.db_dir

    def get_device(self) -> str:
        """
        Get the device used by the vector index client.

        :return: Device name.
        """
        return self.vector_index_client.device

    def get_batch_size(self) -> int:
        """
        Get the batch size used by the embedder.

        :return: Batch size.
        """
        return self.vector_index_client.embedder_batch_size

    def get_max_length(self) -> int | None:
        """
        Get the maximum sequence length for embeddings.

        :return: Maximum length or None if not set.
        """
        return self.vector_index_client.embedder_max_length

    def get_dump_dir(self) -> Path | None:
        """
        Get the directory for saving dumped modules.

        :return: Path to the dump directory or None if dumping is disabled.
        """
        if self.logging_config.dump_modules:
            return self.logging_config.dump_dir
        return None

    def is_multilabel(self) -> bool:
        """
        Check if the dataset is configured for multilabel classification.

        :return: True if multilabel classification is enabled, False otherwise.
        """
        return self.data_handler.multilabel

    def get_n_classes(self) -> int:
        """
        Get the number of classes in the dataset.

        :return: Number of classes.
        """
        return self.data_handler.n_classes

    def is_ram_to_clear(self) -> bool:
        """
        Check if RAM clearing is enabled in the logging configuration.

        :return: True if RAM clearing is enabled, False otherwise.
        """
        return self.logging_config.clear_ram

    def has_saved_modules(self) -> bool:
        """
        Check if any modules have been saved.

        :return: True if there are saved modules, False otherwise.
        """
        node_types = ["regexp", "retrieval", "scoring", "prediction"]
        return any(len(self.optimization_info.modules.get(nt)) > 0 for nt in node_types)
