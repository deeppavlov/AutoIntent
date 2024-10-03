from dataclasses import dataclass


@dataclass
class DNNCScorerConfig:
    _target_: str = "modules.scoring.DNNCScorer"
    model_name: str
    k: int
    train_head: bool
