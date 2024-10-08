from dataclasses import dataclass

from omegaconf import MISSING

from autointent.configs.modules.base import ModuleConfig


@dataclass
class VectorDBConfig(ModuleConfig):
    k: int = MISSING
    model_name: str = MISSING
    _target_: str = "autointent.modules.retrieval.VectorDBModule"
