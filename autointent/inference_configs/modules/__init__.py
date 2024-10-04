from .prediction import ArgmaxPredictorConfig, JinoosPredictorConfig, ThresholdPredictorConfig, TunablePredictorConfig
from .retrieval import VectorDBConfig
from .scoring import DNNCScorerConfig, KNNScorerConfig, LinearScorerConfig, MLKnnScorerConfig
from .search_space import SearchSpace, create_search_space_config

ArgmaxPredictorSearchSpace = create_search_space_config(ArgmaxPredictorConfig, module_type="argmax")
JinoosPredictorSearchSpace = create_search_space_config(JinoosPredictorConfig, module_type="jinoos")
ThresholdPredictorSearchSpace = create_search_space_config(ThresholdPredictorConfig, module_type="threshold")
TunablePredictorSearchSpace = create_search_space_config(TunablePredictorConfig, module_type="tunable")

VectorDBSearchSpace = create_search_space_config(VectorDBConfig, module_type="vector_db")

DNNCScorerSearchSpace = create_search_space_config(DNNCScorerConfig, module_type="dnnc")
KNNScorerSearchSpace = create_search_space_config(KNNScorerConfig, module_type="knn")
LinearScorerSearchSpace = create_search_space_config(LinearScorerConfig, module_type="linear")
MLKnnScorerSearchSpace = create_search_space_config(MLKnnScorerConfig, module_type="mlknn")
