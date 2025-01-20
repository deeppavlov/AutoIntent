"""Configuration for the optimization process."""

from pathlib import Path

from pydantic import BaseModel, Field

from ._name import get_run_name


class DataConfig(BaseModel):
    """Configuration for the data used in the optimization process."""

    train_path: str | Path
    """Path to the training data. Can be local path or HF repo."""


class TaskConfig(BaseModel):
    """Configuration for the task to optimize."""

    search_space_path: Path | None = None
    """Path to the search space configuration file. If None, the default search space will be used"""


class LoggingConfig(BaseModel):
    """Configuration for the logging."""

    project_dir: Path = Field(default_factory=lambda: Path.cwd() / "runs")
    """Path to the directory with different runs."""
    run_name: str = Field(default_factory=get_run_name)
    """Name of the run. If None, a random name will be generated"""
    dump_modules: bool = False
    """Whether to dump the modules or not"""
    clear_ram: bool = False
    """Whether to clear the RAM after dumping the modules"""
    report_to: list[str] | None = None
    """List of callbacks to report to. If None, no callbacks will be used"""

    @property
    def dirpath(self) -> Path:
        """Path to the directory where the logs will be saved."""
        if not hasattr(self, "_dirpath"):
            self._dirpath = self.project_dir / self.run_name
        return self._dirpath

    @property
    def dump_dir(self) -> Path:
        """Path to the directory where the modules will be dumped."""
        if not hasattr(self, "_dump_dir"):
            self._dump_dir = self.dirpath / "modules_dumps"
        return self._dump_dir


class VectorIndexConfig(BaseModel):
    """Configuration for the vector index."""

    save_db: bool = False
    """Whether to save the vector index database or not"""


class TransformerConfig(BaseModel):
    """
    Base class for configuration for the transformer.

    Transformer is used under the hood in :py:class:`autointent.Embedder` and :py:class:`autointent.Ranker`.
    """

    batch_size: int = 32
    """Batch size for the embedder"""
    max_length: int | None = None
    """Max length for the embedder. If None, the max length will be taken from model config"""
    device: str = "cpu"
    """Device to use for the vector index. Can be 'cpu', 'cuda', 'cuda:0', 'mps', etc."""


class EmbedderConfig(TransformerConfig):
    """
    Configuration for the embedder.

    The embedder is used to embed the data before training the model. These parameters
    will be applied to the embedder used in the optimization process in vector db.
    Only one model can be used globally.
    """

    use_cache: bool = True
    """Whether to cache embeddings for reuse, improving performance in repeated operations."""


class CrossEncoderConfig(TransformerConfig):
    """
    Configuration for the embedder.

    The embedder is used to embed the data before training the model. These parameters
    will be applied to the embedder used in the optimization process in vector db.
    Only one model can be used globally.
    """

    train_head: bool = False
    """Whether to train the ranking head of a cross encoder."""
