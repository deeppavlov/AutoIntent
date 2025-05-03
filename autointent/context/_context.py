"""Context manager for configuring and managing data handling, vector indexing, and optimization."""

import logging
from pathlib import Path

import yaml

from autointent import Dataset
from autointent._callbacks import CallbackHandler, get_callbacks
from autointent.configs import CrossEncoderConfig, DataConfig, EmbedderConfig, LoggingConfig

from .data_handler import DataHandler
from .optimization_info import OptimizationInfo


class Context:
    """Context manager for configuring and managing data handling, vector indexing, and optimization.

    This class provides methods to set up logging, configure data and vector index components,
    manage datasets, and retrieve various configurations for inference and optimization.
    Not intended to be instantiated by user.
    """

    data_handler: DataHandler
    """Convenient wrapper for :py:class:`autointent.Dataset`."""

    optimization_info: OptimizationInfo
    """Object for storing optimization trials and inter-node communication."""

    callback_handler = CallbackHandler()
    """Internal callback for logging to tensorboard or wandb."""

    def __init__(self, seed: int | None = 42) -> None:
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
            config: configuration for the transformers to use during optimization.
        """
        if isinstance(config, EmbedderConfig):
            self.embedder_config = config
        elif isinstance(config, CrossEncoderConfig):
            self.cross_encoder_config = config

    def set_dataset(self, dataset: Dataset, config: DataConfig) -> None:
        """Set the datasets for training, validation and testing.

        Args:
            dataset: dataset to use during optimization.
            config: data configuration settings.
        """
        self.data_handler = DataHandler(dataset=dataset, random_seed=self.seed, config=config)

    def dump_optimization_info(self) -> None:
        """Save optimization info to disk."""
        self.optimization_info.dump(self.logging_config.dirpath)

    def dump(self) -> None:
        """Save all information about optimization process to disk.

        Save metrics, hyperparameters, inference, configurations, and datasets to disk.
        """
        self._logger.debug("dumping logs...")
        logs_dir = self.logging_config.dirpath

        self.dump_optimization_info()
        self.data_handler.dataset.to_json(logs_dir / "dataset.json")

        self._logger.info("logs and other assets are saved to %s", logs_dir)

        inference_config = self.optimization_info.get_inference_nodes_config(asdict=True)
        inference_config_path = logs_dir / "inference_config.yaml"
        with inference_config_path.open("w") as file:
            yaml.dump(inference_config, file)

    def load_optimization_info(self) -> None:
        """Restore the context state to resume the optimization process.

        Raises:
            RuntimeError: If the modules artifacts are not found.
        """
        self._logger.debug("loading logs...")
        logs_dir = self.logging_config.dirpath
        self.optimization_info.load(logs_dir)
        if not self.optimization_info.artifacts.has_artifacts():
            msg = (
                "It is impossible to continue from the previous point, "
                "start again with dump_modules=True settings if you want to resume the run."
                "To load optimization info only, use Context.optimization_info.load(logs_dir)."
            )
            raise RuntimeError(msg)

    def get_dump_dir(self) -> Path | None:
        """Get the directory for saving dumped modules.

        Return path to the dump directory or None if dumping is disabled.
        """
        if self.logging_config.dump_modules:
            return self.logging_config.dump_dir
        return None

    def is_multilabel(self) -> bool:
        """Check if the dataset is configured for multilabel classification."""
        return self.data_handler.multilabel

    def is_ram_to_clear(self) -> bool:
        """Check if RAM clearing is enabled in the logging configuration."""
        return self.logging_config.clear_ram

    def has_saved_modules(self) -> bool:
        """Check if any modules have been saved in RAM."""
        node_types = ["regex", "embedding", "scoring", "decision"]
        return any(self.optimization_info.modules.get(nt) is not None for nt in node_types)

    def resolve_embedder(self) -> EmbedderConfig:
        """Resolve the embedder configuration.

        Returns the best embedder configuration or default configuration.

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

        Returns default config if set.

        Raises:
            RuntimeError: If cross-encoder configuration cannot be resolved.
        """
        if hasattr(self, "cross_encoder_config"):
            return self.cross_encoder_config
        msg = "Cross-encoder could't be resolved. Set default config with Context.configure_transformer."
        raise RuntimeError(msg)
