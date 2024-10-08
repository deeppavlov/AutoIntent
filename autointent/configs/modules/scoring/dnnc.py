from dataclasses import dataclass

from omegaconf import MISSING

from autointent.configs.modules.base import ModuleConfig


@dataclass
class DNNCScorerConfig(ModuleConfig):
    model_name: str = MISSING
    k: int = MISSING
    train_head: bool = False
    _target_: str = "autointent.modules.scoring.DNNCScorer"
