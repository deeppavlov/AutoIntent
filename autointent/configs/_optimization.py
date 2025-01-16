"""Configuration for the optimization process."""

from dataclasses import dataclass, field
from pathlib import Path

from ._name import get_run_name


@dataclass
class DataConfig:
    """Configuration for the data used in the optimization process."""

    train_path: str | Path
    """Path to the training data. Can be local path or HF repo."""
    test_path: Path | None = None
    """Path to the testing data. If None, no testing data will be used"""
    force_multilabel: bool = False
    """Force multilabel classification even if the data is multiclass"""


@dataclass
class TaskConfig:
    """Configuration for the task to optimize."""

    search_space_path: Path | None = None
    """Path to the search space configuration file. If None, the default search space will be used"""


@dataclass
class LoggingConfig:
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

    def __post_init__(self) -> None:
        """Define the run name, directory path and dump directory."""
        self.define_run_name()
        self.define_dirpath()
        self.define_dump_dir()

    def define_run_name(self) -> None:
        """Define the run name. If None, a random name will be generated."""
        self.run_name = get_run_name(self.run_name)

    def define_dirpath(self) -> None:
        """Define the directory path. If None, the logs will be saved in the current working directory."""
        dirpath = Path.cwd() / "runs" if self.dirpath is None else self.dirpath
        if self.run_name is None:
            raise ValueError
        self.dirpath = dirpath / self.run_name

    def get_dirpath(self) -> Path:
        """Get the directory path."""
        if self.dirpath is None:
            raise ValueError
        return self.dirpath

    def get_run_name(self) -> str:
        """Get the run name."""
        if self.run_name is None:
            raise ValueError
        return self.run_name

    def define_dump_dir(self) -> None:
        """Define the dump directory. If None, the modules will not be dumped."""
        if self.dump_dir is None:
            if self.dirpath is None:
                raise ValueError
            self.dump_dir = self.dirpath / "modules_dumps"


@dataclass
class VectorIndexConfig:
    """Configuration for the vector index."""

    save_db: bool = False
    """Whether to save the vector index database or not"""


@dataclass
class TransformerConfig:
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


@dataclass
class EmbedderConfig(TransformerConfig):
    """
    Configuration for the embedder.

    The embedder is used to embed the data before training the model. These parameters
    will be applied to the embedder used in the optimization process in vector db.
    Only one model can be used globally.
    """

    use_cache: bool = True
    """Whether to cache embeddings for reuse, improving performance in repeated operations."""


@dataclass
class CrossEncoderConfig(TransformerConfig):
    """
    Configuration for the embedder.

    The embedder is used to embed the data before training the model. These parameters
    will be applied to the embedder used in the optimization process in vector db.
    Only one model can be used globally.
    """

    train_head: bool = False
    """Whether to train the ranking head of a cross encoder."""


@dataclass
class OptimizationConfig:
    """Configuration for the optimization process."""

    seed: int = 0
    """Seed for the random number generator"""
    data: DataConfig = field(default_factory=DataConfig)
    """Configuration for the data used in the optimization process"""
    task: TaskConfig = field(default_factory=TaskConfig)
    """Configuration for the task to optimize"""
    logs: LoggingConfig = field(default_factory=LoggingConfig)
    """Configuration for the logging"""
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)
    """Configuration for the vector index"""
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    """Configuration for the embedder"""
    cross_encoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    """Configuration for the cross encoder"""
