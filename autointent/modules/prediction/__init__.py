from .argmax import ArgmaxPredictor
from .base import PredictionModule
from .jinoos import JinoosPredictor
from .logit_adaptivness import LogitAdaptivnessPredictor
from .threshold import ThresholdPredictor
from .tunable import TunablePredictor

__all__ = ["ArgmaxPredictor",
           "JinoosPredictor",
           "PredictionModule",
           "ThresholdPredictor",
           "TunablePredictor",
           "LogitAdaptivnessPredictor"]
