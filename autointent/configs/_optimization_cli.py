"""Configuration for the optimization process."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from ._name import get_run_name


@dataclass
class DataConfig:
    """Configuration for the data used in the optimization process."""

    train_path: str | Path = MISSING
    """Path to the training data"""
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
    clear_ram: bool = True
    """Whether to clear the RAM after dumping the modules"""

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

    def define_dump_dir(self) -> None:
        """Define the dump directory. If None, the modules will not be dumped."""
        if self.dump_dir is None:
            if self.dirpath is None:
                raise ValueError
            self.dump_dir = self.dirpath / "modules_dumps"


@dataclass
class VectorIndexConfig:
    """Configuration for the vector index."""

    db_dir: Path | None = None
    """Path to the directory where the vector index database will be saved. If None, the database will not be saved"""
    device: str = "cpu"
    """Device to use for the vector index. Can be 'cpu', 'cuda', 'cuda:0', 'mps', etc."""
    save_db: bool = False
    """Whether to save the vector index database or not"""


@dataclass
class AugmentationConfig:
    """Configuration for the augmentation."""

    regex_sampling: int = 0
    """Number of regex samples to generate"""
    multilabel_generation_config: str | None = None
    """Path to the multilabel generation configuration file. If None, the default configuration will be used"""


@dataclass
class EmbedderConfig:
    """
    Configuration for the embedder.

    The embedder is used to embed the data before training the model. These parameters
    will be applied to the embedder used in the optimization process in vector db.
    Only one model can be used globally.
    """

    batch_size: int = 32
    """Batch size for the embedder"""
    max_length: int | None = None
    """Max length for the embedder. If None, the max length will be taken from model config"""


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

    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"override hydra/job_logging": "autointent_standard_job_logger"},
            {"override hydra/help": "autointent_help"},
        ],
    )


logger_config = {
    "version": 1,
    "formatters": {"simple": {"format": "%(asctime)s - %(name)s [%(levelname)s] %(message)s"}},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "simple",
            "filename": "${hydra.runtime.output_dir}/${hydra.job.name}.log",
        },
    },
    "root": {"level": "WARN", "handlers": ["console", "file"]},
    "disable_existing_loggers": "false",
}

help_config = {
    "app_name": "AutoIntent",
    "header": "== ${hydra.help.app_name} ==",
    "footer": """
Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help""",
    "template": """
  ${hydra.help.header}

  This is ${hydra.help.app_name}!
  == Config ==
  This is the config generated for this run.
  You can override everything, for example:
  python my_app.py db.user=foo db.pass=bar
  -------
  $CONFIG
  -------

  ${hydra.help.footer}""",
}


cs = ConfigStore.instance()
cs.store(name="optimization_config", node=OptimizationConfig)
cs.store(name="autointent_standard_job_logger", group="hydra/job_logging", node=logger_config)
cs.store(name="autointent_help", group="hydra/help", node=help_config)
