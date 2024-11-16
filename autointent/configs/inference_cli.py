from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from autointent.custom_types import LogLevel


@dataclass
class InferenceConfig:
    """Configuration for the inference process."""

    data_path: str
    """Path to the file containing the data for prediction"""
    source_dir: str
    """Path to the directory containing the inference config"""
    output_path: str
    """Path to the file where the predictions will be saved"""
    log_level: LogLevel = LogLevel.ERROR
    """Logging level"""
    with_metadata: bool = False
    """Whether to save metadata along with the predictions"""


cs = ConfigStore.instance()
cs.store(name="inference_config", node=InferenceConfig)
