"""Context manager for configuring and managing data handling, vector indexing, and optimization."""

import logging
from pathlib import Path

import yaml
from typing_extensions import assert_never

from autointent import Dataset
from autointent._callbacks import CallbackHandler, get_callbacks
from autointent.configs import CrossEncoderConfig, DataConfig, EmbedderConfig, HFModelConfig, HPOConfig, LoggingConfig

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
        if isinstance(config, LoggingConfig):
            self.logging_config = config
            self.callback_handler = get_callbacks(config.report_to)
            self.optimization_info = OptimizationInfo()
        else:
            assert_never(config)

    def configure_transformer(self, config: EmbedderConfig | CrossEncoderConfig | HFModelConfig) -> None:
        """Configure the vector index client and embedder.

        Args:
            config: configuration for the transformers to use during optimization.
        """
        if isinstance(config, EmbedderConfig):
            self.embedder_config = config
        elif isinstance(config, CrossEncoderConfig):
            self.cross_encoder_config = config
        elif isinstance(config, HFModelConfig):
            self.transformer_config = config
        else:
            assert_never(config)

    def configure_hpo(self, config: HPOConfig) -> None:
        if isinstance(config, HPOConfig):
            self.hpo_config = config
        else:
            assert_never(config)

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

        This method returns the configuration with the following priorities:
        - the best embedder configuration obtained during embedding node optimization
        - default configuration preset by user with :py:meth:`Context.configure_transformer`
        - default configuration preset by AutoIntent in :py:class:`autointent.configs.EmbedderConfig`
        """
        try:
            return self.optimization_info.get_best_embedder()
        except ValueError:
            if hasattr(self, "embedder_config"):
                return self.embedder_config
            return EmbedderConfig()

    def resolve_ranker(self) -> CrossEncoderConfig:
        """Resolve the cross-encoder configuration.

        This method returns the configuration with the following priorities:
        - default configuration preset by user with :py:meth:`Context.configure_transformer`
        - default configuration preset by AutoIntent in :py:class:`autointent.configs.CrossEncoderConfig`
        """
        if hasattr(self, "cross_encoder_config"):
            return self.cross_encoder_config
        return CrossEncoderConfig()

    def resolve_transformer(self) -> HFModelConfig:
        """Resolve the transformer configuration.

        This method returns the configuration with the following priorities:
        - the best transformer configuration obtained during embedding node optimization
        - default configuration preset by user with :py:meth:`Context.configure_transformer`
        - default configuration preset by AutoIntent in :py:class:`autointent.configs.HFModelConfig`
        """
        try:
            return self.optimization_info.get_best_embedder()
        except ValueError:
            if hasattr(self, "transformer_config"):
                return self.transformer_config
            return HFModelConfig()
