from .base import ModuleConfig
from .prediction import ArgmaxPredictorConfig, JinoosPredictorConfig, ThresholdPredictorConfig, TunablePredictorConfig
from .retrieval import VectorDBConfig
from .scoring import DNNCScorerConfig, KNNScorerConfig, LinearScorerConfig, MLKnnScorerConfig

PREDICTION_MODULES_CONFIGS: dict[str, type[ModuleConfig]] = {
    "argmax": ArgmaxPredictorConfig,
    "jinoos": JinoosPredictorConfig,
    "threshold": ThresholdPredictorConfig,
    "tunable": TunablePredictorConfig,
}

RETRIEVAL_MODULES_CONFIGS: dict[str, type[ModuleConfig]] = {"vector_db": VectorDBConfig}

SCORING_MODULES_CONFIGS: dict[str, type[ModuleConfig]] = {
    "dnnc": DNNCScorerConfig,
    "knn": KNNScorerConfig,
    "linear": LinearScorerConfig,
    "mlknn": MLKnnScorerConfig,
}

MODULES_CONFIGS: dict[str, dict[str, type[ModuleConfig]]] = {
    "retrieval": RETRIEVAL_MODULES_CONFIGS,
    "scoring": SCORING_MODULES_CONFIGS,
    "prediction": PREDICTION_MODULES_CONFIGS,
}
