from dataclasses import dataclass

from autointent.configs.modules.base import ModuleConfig


@dataclass
class VectorDBConfig(ModuleConfig):
    _target_: str = "modules.retrieval.VectorDBModule"
    k: int
    model_name: str
