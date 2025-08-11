"""These modules take predicted probabilities and apply some decision rule to define final set of predicted labels."""

from ._adaptive import AdaptiveDecision
from ._argmax import ArgmaxDecision
from ._jinoos import JinoosDecision
from ._threshold import ThresholdDecision
from ._tunable import TunableDecision

__all__ = [
    "AdaptiveDecision",
    "ArgmaxDecision",
    "JinoosDecision",
    "ThresholdDecision",
    "TunableDecision",
]
