from dataclasses import dataclass

from autointent.configs.modules.base import ModuleConfig


@dataclass
class DNNCScorerConfig(ModuleConfig):
    _target_: str = "modules.scoring.DNNCScorer"
    model_name: str
    k: int
    train_head: bool
