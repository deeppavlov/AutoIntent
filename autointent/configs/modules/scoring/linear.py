from dataclasses import dataclass

from autointent.configs.modules.base import ModuleConfig


@dataclass
class LinearScorerConfig(ModuleConfig):
    _target_: str = "autointent.modules.scoring.LinearScorer"
