"""Configuration for the optimization process."""

from pathlib import Path

from pydantic import BaseModel, Field, ValidationInfo, field_validator

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

    @field_validator("run_name", mode="before")
    @classmethod
    def define_run_name(cls, v: str | None) -> str:
        """Define the run name. If None, a random name will be generated."""
        return get_run_name(v)

    @field_validator("dirpath", mode="before")
    @classmethod
    def define_dirpath(cls, v: Path | None, info: ValidationInfo) -> Path:
        """Define the directory path. If None, the logs will be saved in the current working directory."""
        if v is None:
            v = Path.cwd() / "runs"
        return v / str(info.data["run_name"])

    @field_validator("dump_dir", mode="before")
    @classmethod
    def define_dump_dir(cls, v: Path | None, info: ValidationInfo) -> Path:
        """Define the dump directory. If None, the modules will not be dumped."""
        if v is None:
            v = info.data["dirpath"] / "modules_dumps"
        return v

    @property
    def safe_run_name(self) -> str:
        # This property ensures that the type checker knows `run_name` is a `str`
        if self.run_name is None:
            msg = "run_name should not be None after validation"
            raise ValueError(msg)
        return self.run_name

    @property
    def safe_dirpath(self) -> Path:
        # This property ensures that the type checker knows `run_name` is a `str`
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


class OptimizationConfig(BaseModel):
    """Configuration for the optimization process."""

    data: DataConfig
    """Configuration for the data used in the optimization process"""
    task: TaskConfig = Field(default_factory=TaskConfig)
    """Configuration for the task to optimize"""
    logs: LoggingConfig = Field(default_factory=LoggingConfig)
    """Configuration for the logging"""
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    """Configuration for the vector index"""
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    """Configuration for the embedder"""
    cross_encoder: CrossEncoderConfig = Field(default_factory=CrossEncoderConfig)
    """Configuration for the cross encoder"""
    seed: int = 0
    """Seed for the random number generator"""
