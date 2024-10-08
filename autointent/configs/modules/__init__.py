from .base import ModuleConfig
from .prediction import ArgmaxPredictorConfig, JinoosPredictorConfig, ThresholdPredictorConfig, TunablePredictorConfig
from .retrieval import VectorDBConfig
from .scoring import DNNCScorerConfig, KNNScorerConfig, LinearScorerConfig, MLKnnScorerConfig
from .search_space import SearchSpaceDataclass, create_search_space_model

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

MODULES_CONFIGS = {
    "retrieval": RETRIEVAL_MODULES_CONFIGS,
    "scoring": SCORING_MODULES_CONFIGS,
    "prediction": PREDICTION_MODULES_CONFIGS,
}

# from hydra.core.config_store import ConfigStore
# config_groups = [PREDICTION_MODULES_CONFIGS, RETRIEVAL_MODULES_CONFIGS, SCORING_MODULES_CONFIGS]
# groups_names = ["prediction_modules_configs", "retrieval_modules_configs", "scoring_modules_configs"]

# cs = ConfigStore.instance()
# for config_group, group_name in zip(config_groups, groups_names, strict=True):
#     for module_type, module_config in config_group.items():
#         cs.store(
#             name=module_type,
#             node=module_config,
#             group=group_name,
#         )
