from dataclasses import dataclass


@dataclass
class VectorDBConfig:
    _target_: str = "modules.retrieval.VectorDBModule"
    k: int
    model_name: str
