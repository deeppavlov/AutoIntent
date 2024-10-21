from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from autointent.custom_types import LogLevel


@dataclass
class InferenceConfig:
    data_path: str
    source_dir: str
    output_path: str
    log_level: LogLevel = LogLevel.ERROR


cs = ConfigStore.instance()
cs.store(name="inference_config", node=InferenceConfig)
