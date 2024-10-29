from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from autointent.custom_types import LogLevel
from autointent.pipeline.optimization.utils import generate_name


@dataclass
class DataConfig:
    train_path: str | Path = MISSING
    test_path: Path | None = None
    force_multilabel: bool = False


@dataclass
class TaskConfig:
    """TODO presets"""

    search_space_path: Path | None = None


@dataclass
class LoggingConfig:
    run_name: str | None = None
    dirpath: Path | None = None
    level: LogLevel = LogLevel.ERROR
    dump_dir: Path | None = None

    def __post_init__(self) -> None:
        self.define_run_name()
        self.define_dirpath()
        self.define_dump_dir()

    def define_run_name(self) -> None:
        if self.run_name is None:
            self.run_name = generate_name()
        self.run_name = f"{self.run_name}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}"  # noqa: DTZ005

    def define_dirpath(self) -> None:
        dirpath = Path.cwd() / "runs" if self.dirpath is None else self.dirpath
        if self.run_name is None:
            raise ValueError
        self.dirpath = dirpath / self.run_name
        self.dirpath.mkdir(parents=True)

    def define_dump_dir(self) -> None:
        if self.dump_dir is None:
            if self.dirpath is None:
                raise ValueError
            self.dump_dir = self.dirpath / "modules_dumps"


@dataclass
class VectorIndexConfig:
    db_dir: Path | None = None
    device: str = "cpu"


@dataclass
class AugmentationConfig:
    regex_sampling: int = 0
    multilabel_generation_config: str | None = None


@dataclass
class EmbedderConfig:
    batch_size: int = 1
    max_length: int | None = None


@dataclass
class OptimizationConfig:
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    logs: LoggingConfig = field(default_factory=LoggingConfig)
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)

    defaults: list[Any] = field(
        default_factory=lambda: ["_self_", {"override hydra/job_logging": "autointent_standard_job_logger"}]
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


cs = ConfigStore.instance()
cs.store(name="optimization_config", node=OptimizationConfig)
cs.store(name="autointent_standard_job_logger", group="hydra/job_logging", node=logger_config)
