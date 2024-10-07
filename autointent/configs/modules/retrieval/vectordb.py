from dataclasses import dataclass


@dataclass
class VectorDBConfig:
    k: int
    model_name: str
    _target_: str = "autointent.modules.retrieval.VectorDBModule"
