from dataclasses import dataclass
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from autointent.custom_types import LogLevel


@dataclass
class OptimizationConfig:
    dataset_path: Path = MISSING
    search_space_path: Path | None = None
    test_path: Path | None = None
    db_dir: Path | None = None
    logs_dir: Path | None = None
    run_name: str | None = None
    device: str = "cpu"
    regex_sampling: int = 0
    seed: int = 0
    log_level: LogLevel = LogLevel.ERROR
    multilabel_generation_config: str | None = None
    force_multilabel: bool = False


cs = ConfigStore.instance()
cs.store(name="optimization_config", node=OptimizationConfig)
