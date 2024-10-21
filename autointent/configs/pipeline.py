from dataclasses import dataclass
from enum import Enum

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from autointent.custom_types import LogLevel

from .node import NodeOptimizerConfig


@dataclass
class PipelineSearchSpace:
    nodes: list[NodeOptimizerConfig] = MISSING
    _target_: str = "autointent.pipeline.pipeline.Pipeline"


class ClassificationMode(Enum):
    multiclass = "multiclass"
    multilabel = "multilabel"
    multiclass_as_multilabel = "multiclass_as_multilabel"


@dataclass
class OptimizationConfig:
    multiclass_path: str = MISSING
    search_space_path: str | None = None
    multilabel_path: str | None = None
    test_path: str | None = None
    db_dir: str | None = None
    logs_dir: str | None = None
    run_name: str | None = None
    mode: ClassificationMode = ClassificationMode.multiclass
    device: str | None = None
    regex_sampling: int = 0
    seed: int = 0
    log_level: LogLevel = LogLevel.ERROR
    multilabel_generation_config: str | None = None


cs = ConfigStore.instance()
cs.store(name="optimization_config", node=OptimizationConfig)
