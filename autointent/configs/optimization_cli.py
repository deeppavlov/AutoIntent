from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

from autointent.custom_types import LogLevel


@dataclass
class OptimizationConfig:
    search_space_path: str = ""
    dataset_path: str = ""
    test_path: str = ""
    db_dir: str = ""
    logs_dir: str = "./outputs"
    run_name: str = ""
    device: str = "cuda:0"
    regex_sampling: int = 0
    seed: int = 0
    log_level: LogLevel = LogLevel.ERROR
    multilabel_generation_config: str = ""
    force_multilabel: bool = False
    embedder_batch_size: int = 1
    embedder_max_length: int | None = None

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
    "root": {"level": "INFO", "handlers": ["console", "file"]},
    "disable_existing_loggers": "false",
}


cs = ConfigStore.instance()
cs.store(name="optimization_config", node=OptimizationConfig)
cs.store(name="autointent_standard_job_logger", group="hydra/job_logging", node=logger_config)
