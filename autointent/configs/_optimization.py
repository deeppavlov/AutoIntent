"""Configuration for the optimization process."""

from pathlib import Path

from pydantic import BaseModel, model_validator

from ._name import get_run_name


class DataConfig(BaseModel):
    """Configuration for the data used in the optimization process."""

    train_path: str | Path
    """Path to the training data. Can be local path or HF repo."""
    test_path: Path | None = None
    """Path to the testing data. If None, no testing data will be used"""
    force_multilabel: bool = False
    """Force multilabel classification even if the data is multiclass"""


class TaskConfig(BaseModel):
    """Configuration for the task to optimize."""

    search_space_path: Path | None = None
    """Path to the search space configuration file. If None, the default search space will be used"""


class LoggingConfig(BaseModel):
    """Configuration for the logging."""

    run_name: str | None = None
    """Name of the run. If None, a random name will be generated"""
    dirpath: Path | None = None
    """Path to the directory where the logs will be saved.
    If None, the logs will be saved in the current working directory"""
    dump_dir: Path | None = None
    """Path to the directory where the modules will be dumped. If None, the modules will not be dumped"""
    dump_modules: bool = False
    """Whether to dump the modules or not"""
    clear_ram: bool = False
    """Whether to clear the RAM after dumping the modules"""
    report_to: list[str] | None = None
    """List of callbacks to report to. If None, no callbacks will be used"""

    @model_validator(mode="after")
    def fill_nones(self) -> "LoggingConfig":
        self.define_run_name()
        self.define_dirpath()
        self.define_dump_dir()
        return self

    def define_run_name(self) -> None:
        """Define the run name. If None, a random name will be generated."""
        self.run_name = get_run_name(self.run_name)

    def define_dirpath(self) -> None:
        """Define the directory path. If None, the logs will be saved in the current working directory."""
        dirpath = Path.cwd() / "runs" if self.dirpath is None else self.dirpath
        if self.run_name is None:
            raise ValueError
        self.dirpath = dirpath / self.run_name

    def define_dump_dir(self) -> None:
        """Define the dump directory. If None, the modules will not be dumped."""
        if self.dump_dir is None:
            if self.dirpath is None:
                raise ValueError
            self.dump_dir = self.dirpath / "modules_dumps"

    @property
    def safe_run_name(self) -> str:
        """Use this method for type safety instead of :py:attr:`LoggingConfig.run_name`."""
        if self.run_name is None:
            msg = "run_name should not be None after validation"
            raise ValueError(msg)
        return self.run_name

    @property
    def safe_dirpath(self) -> Path:
        """Use this method for type safety instead of :py:attr:`LoggingConfig.dirpath`."""
        if self.dirpath is None:
            msg = "dirpath should not be None after validation"
            raise ValueError(msg)
        return self.dirpath


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
