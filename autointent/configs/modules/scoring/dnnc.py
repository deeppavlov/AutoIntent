from dataclasses import dataclass


@dataclass
class DNNCScorerConfig:
    model_name: str
    k: int
    train_head: bool
    _target_: str = "autointent.modules.scoring.DNNCScorer"
