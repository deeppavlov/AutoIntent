from dataclasses import dataclass
from enum import Enum

from hydra.core.config_store import ConfigStore

from autointent.custom_types import LogLevel


class ClassificationMode(Enum):
    multiclass = "multiclass"
    multilabel = "multilabel"
    multiclass_as_multilabel = "multiclass_as_multilabel"


@dataclass
class OptimizationConfig:
    search_space_path: str = ""
    multiclass_path: str = ""
    multilabel_path: str = ""
    test_path: str = ""
    db_dir: str = ""
    logs_dir: str = ""
    run_name: str = ""
    mode: ClassificationMode = ClassificationMode.multiclass
    device: str = "cuda:0"
    regex_sampling: int = 0
    seed: int = 0
    log_level: LogLevel = LogLevel.ERROR
    multilabel_generation_config: str = ""


cs = ConfigStore.instance()
cs.store(name="optimization_config", node=OptimizationConfig)
