import numpy as np

from autointent import Context
from autointent.modules import ThresholdPredictor, VectorDBModule
from autointent.modules.prediction.base import get_prediction_evaluation_data
from autointent.metrics import retrieval_hit_rate, scoring_roc_auc


def get_fit_data(context: Context):
    retrieval_params = {"k": 3, "model_name": "sergeyzh/rubert-tiny-turbo"}
    vector_db = VectorDBModule(**retrieval_params)
    vector_db.fit(context.data_handler.utterances_train, context.data_handler.labels_train)
    metric_value = vector_db.score(context, retrieval_hit_rate)
    artifact = vector_db.get_assets()
    context.optimization_info.log_module_optimization(
        node_type="scoring",
        module_type="vector_db",
        module_params=retrieval_params,
        metric_value=metric_value,
        metric_name="retrieval_hit_rate_macro",
        artifact=artifact,
        module_dump_dir="",
    )
    labels, scores = get_prediction_evaluation_data(context)
    args = (scores, labels, context.data_handler.tags)  # type: ignore[assignment]
    return args


def test_predict_returns_correct_indices(context):
    predictor = ThresholdPredictor(0.5)
    predictor.fit(*get_fit_data(context("multiclass")))
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))


def test_predict_returns_list(context):
    predictor = ThresholdPredictor(np.array([0.5, 0.5, 0.5]))
    predictor.fit(*get_fit_data(context("multiclass")))
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))


def test_predict_handles_single_class(context):
    predictor = ThresholdPredictor(0.5)
    predictor.fit(*get_fit_data(context("multiclass")))
    scores = np.array([[0.5], [0.5], [0.5]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([0, 0, 0]))
