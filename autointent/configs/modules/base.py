from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ModuleConfig:
    _target_: str = MISSING
