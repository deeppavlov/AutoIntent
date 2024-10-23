from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from autointent.custom_types import LogLevel


@dataclass
class DataConfig:
    train_path: Path = MISSING
    test_path: Path | None = None
    force_multilabel: bool = False


@dataclass
class TaskConfig:
    """TODO presets"""
    search_space_path: Path | None = None


@dataclass
class LoggingConfig:
    dirpath: Path | None = None
    run_name: str | None = None
    level: LogLevel = LogLevel.ERROR


@dataclass
class VectorIndexConfig:
    db_dir: Path | None = None
    device: str = "cpu"


@dataclass
class AugmentationConfig:
    regex_sampling: int = 0
    multilabel_generation_config: str | None = None


@dataclass
class OptimizationConfig:
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    logs: LoggingConfig = field(default_factory=LoggingConfig)
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


cs = ConfigStore.instance()
cs.store(name="optimization_config", node=OptimizationConfig)
