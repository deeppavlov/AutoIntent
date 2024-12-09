"""These modules take predicted probabilities and apply some decision rule to define final set of predicted labels."""

from ._adaptive import AdaptivePredictor
from ._argmax import ArgmaxPredictor
from ._jinoos import JinoosPredictor
from ._threshold import ThresholdPredictor
from ._tunable import TunablePredictor

__all__ = [
    "AdaptivePredictor",
    "ArgmaxPredictor",
    "JinoosPredictor",
    "PredictionModule",
    "ThresholdPredictor",
    "TunablePredictor",
]
