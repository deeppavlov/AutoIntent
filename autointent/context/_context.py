"""Context manager for configuring and managing data handling, vector indexing, and optimization."""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from autointent import Dataset
from autointent._callbacks import CallbackHandler, get_callbacks
from autointent.configs import CrossEncoderConfig, DataConfig, EmbedderConfig, LoggingConfig

from ._utils import NumpyEncoder
from .data_handler import DataHandler
from .optimization_info import OptimizationInfo


class Context:
    """Context manager for configuring and managing data handling, vector indexing, and optimization.

    This class provides methods to set up logging, configure data and vector index components,
    manage datasets, and retrieve various configurations for inference and optimization.

    Attributes:
        data_handler: Handler for managing datasets.
        optimization_info: Container for optimization information.
        callback_handler: Handler for managing callbacks.
    """

    data_handler: DataHandler
    optimization_info: OptimizationInfo
    callback_handler = CallbackHandler()

    def __init__(self, seed: int = 42) -> None:
        """Initialize the Context object.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self._logger = logging.getLogger(__name__)

    def configure_logging(self, config: LoggingConfig) -> None:
        """Configure logging settings.

        Args:
            config: Logging configuration settings.
        """
        self.logging_config = config
        self.callback_handler = get_callbacks(config.report_to)
        self.optimization_info = OptimizationInfo()

    def configure_transformer(self, config: EmbedderConfig | CrossEncoderConfig) -> None:
        """Configure the vector index client and embedder.

        Args:
            config: Configuration for the vector index.
        """
        if isinstance(config, EmbedderConfig):
            self.embedder_config = config
        elif isinstance(config, CrossEncoderConfig):
            self.cross_encoder_config = config

    def set_dataset(self, dataset: Dataset, config: DataConfig) -> None:
        """Set the datasets for training, validation and testing.

        Args:
            dataset: Dataset.
            config: Data configuration settings.
        """
        self.data_handler = DataHandler(dataset=dataset, random_seed=self.seed, config=config)

    def get_inference_config(self) -> dict[str, Any]:
        """Generate configuration settings for inference.

        Returns:
            Dictionary containing inference configuration.
        """
        nodes_configs = self.optimization_info.get_inference_nodes_config(asdict=True)
        return {
            "metadata": {
                "multilabel": self.is_multilabel(),
                "n_classes": self.get_n_classes(),
                "seed": self.seed,
            },
            "nodes_configs": nodes_configs,
        }

    def dump(self) -> None:
        """Save logs, configurations, and datasets to disk."""
        self._logger.debug("dumping logs...")
        optimization_results = self.optimization_info.dump_evaluation_results()

        logs_dir = self.logging_config.dirpath
        logs_dir.mkdir(parents=True, exist_ok=True)

        logs_path = logs_dir / "logs.json"
        with logs_path.open("w") as file:
            json.dump(optimization_results, file, indent=4, ensure_ascii=False, cls=NumpyEncoder)

        self.data_handler.dataset.to_json(logs_dir / "dataset.json")

        self._logger.info("logs and other assets are saved to %s", logs_dir)

        inference_config = self.get_inference_config()
        inference_config_path = logs_dir / "inference_config.yaml"
        with inference_config_path.open("w") as file:
            yaml.dump(inference_config, file)

    def get_dump_dir(self) -> Path | None:
        """Get the directory for saving dumped modules.

        Returns:
            Path to the dump directory or None if dumping is disabled.
        """
        if self.logging_config.dump_modules:
            return self.logging_config.dump_dir
        return None

    def is_multilabel(self) -> bool:
        """Check if the dataset is configured for multilabel classification.

        Returns:
            True if multilabel classification is enabled, False otherwise.
        """
        return self.data_handler.multilabel

    def get_n_classes(self) -> int:
        """Get the number of classes in the dataset.

        Returns:
            Number of classes.
        """
        return self.data_handler.n_classes

    def is_ram_to_clear(self) -> bool:
        """Check if RAM clearing is enabled in the logging configuration.

        Returns:
            True if RAM clearing is enabled, False otherwise.
        """
        return self.logging_config.clear_ram

    def has_saved_modules(self) -> bool:
        """Check if any modules have been saved.

        Returns:
            True if there are saved modules, False otherwise.
        """
        node_types = ["regex", "embedding", "scoring", "decision"]
        return any(len(self.optimization_info.modules.get(nt)) > 0 for nt in node_types)

    def resolve_embedder(self) -> EmbedderConfig:
        """Resolve the embedder configuration.

        Returns:
            The best embedder configuration or default configuration.

        Raises:
            RuntimeError: If embedder configuration cannot be resolved.
        """
        try:
            return self.optimization_info.get_best_embedder()
        except ValueError as e:
            if hasattr(self, "embedder_config"):
                return self.embedder_config
            msg = (
                "Embedder could't be resolved. Either include embedding node into the "
                "search space or set default config with Context.configure_transformer."
            )
            raise RuntimeError(msg) from e

    def resolve_ranker(self) -> CrossEncoderConfig:
        """Resolve the cross-encoder configuration.

        Returns:
            The cross-encoder configuration.

        Raises:
            RuntimeError: If cross-encoder configuration cannot be resolved.
        """
        if hasattr(self, "cross_encoder_config"):
            return self.cross_encoder_config
        msg = "Cross-encoder could't be resolved. Set default config with Context.configure_transformer."
        raise RuntimeError(msg)
