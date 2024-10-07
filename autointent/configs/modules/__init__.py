from .base import ModuleConfig
from .prediction import ArgmaxPredictorConfig, JinoosPredictorConfig, ThresholdPredictorConfig, TunablePredictorConfig
from .retrieval import VectorDBConfig
from .scoring import DNNCScorerConfig, KNNScorerConfig, LinearScorerConfig, MLKnnScorerConfig
from .search_space import SearchSpace, create_search_space_config

PREDICTION_MODULES_CONFIGS = {
    "argmax": ArgmaxPredictorConfig,
    "jinoos": JinoosPredictorConfig,
    "threshold": ThresholdPredictorConfig,
    "tunable": TunablePredictorConfig,
}

RETRIEVAL_MODULES_CONFIGS = {
    "vector_db": VectorDBConfig
}

SCORING_MODULES_CONFIGS = {
    "dnnc": DNNCScorerConfig,
    "knn": KNNScorerConfig,
    "linear": LinearScorerConfig,
    "mlknn": MLKnnScorerConfig,
}
