"""Context manager for configuring and managing data handling, vector indexing, and optimization."""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from autointent import Dataset
from autointent.configs import (
    DataConfig,
    EmbedderConfig,
    LoggingConfig,
    VectorIndexConfig,
)

from ._utils import NumpyEncoder, load_data
from .data_handler import DataHandler
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
            self.embedder_config.device,
            self.vector_index_config.db_dir,
            self.embedder_config.batch_size,
            self.embedder_config.max_length,
            self.embedder_config.use_cache,
        )

    def configure_data(self, config: DataConfig) -> None:
        """
        Configure data handling.

        :param config: Configuration for the data handling process.
        """
        self.data_handler = DataHandler(
            dataset=load_data(config.train_path),
            random_seed=self.seed,
            force_multilabel=config.force_multilabel,
        )

    def set_dataset(self, dataset: Dataset, force_multilabel: bool = False) -> None:
        """
        Set the datasets for training, validation and testing.

        :param dataset: Dataset.
        :param force_multilabel: Whether to force multilabel classification.
        """
        self.data_handler = DataHandler(
            dataset=dataset,
            force_multilabel=force_multilabel,
            random_seed=self.seed,
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
        nodes_configs = self.optimization_info.get_inference_nodes_config(asdict=True)
        return {
            "metadata": {
                "embedder_device": self.get_device(),
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

        # self._logger.info(make_report(optimization_results, nodes=nodes))

        # dump train and test data splits
        dataset_path = logs_dir / "dataset.json"
        with dataset_path.open("w") as file:
            json.dump(self.data_handler.dump(), file, indent=4, ensure_ascii=False)

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
        Get the embedder device used by the vector index client.

        :return: Device name.
        """
        return self.vector_index_client.embedder_device

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

    def get_use_cache(self) -> bool:
        """
        Check if caching is enabled for the embedder.

        :return: True if caching is enabled, False otherwise.
        """
        return self.vector_index_client.embedder_use_cache

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
