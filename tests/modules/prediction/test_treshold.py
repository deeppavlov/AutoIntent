import numpy as np

from autointent import Context
from autointent.metrics import retrieval_hit_rate, scoring_roc_auc
from autointent.modules import KNNScorer, ThresholdPredictor, VectorDBModule
from autointent.modules.prediction.base import get_prediction_evaluation_data


def get_fit_data(context: Context, db_dir):
    retrieval_params = {"k": 3, "model_name": "sergeyzh/rubert-tiny-turbo", "db_dir": db_dir}
    vector_db = VectorDBModule(**retrieval_params)
    vector_db.fit(context.data_handler.utterances_train, context.data_handler.labels_train)
    metric_value = vector_db.score(context, retrieval_hit_rate)
    artifact = vector_db.get_assets()
    context.optimization_info.log_module_optimization(
        node_type="retrieval",
        module_type="vector_db",
        module_params=retrieval_params,
        metric_value=metric_value,
        metric_name="retrieval_hit_rate_macro",
        artifact=artifact,
        module_dump_dir="",
    )
    knn_params = {
        "k": 3,
        "weights": "distance",
        "model_name": "sergeyzh/rubert-tiny-turbo",
        "db_dir": db_dir,
        "n_classes": 3,
        "multilabel": False,
    }
    scorer = KNNScorer(**knn_params)

    scorer.fit(context.data_handler.utterances_train, context.data_handler.labels_train)
    score = scorer.score(context, scoring_roc_auc)
    assets = scorer.get_assets()
    context.optimization_info.log_module_optimization(
        "scoring",
        "knn",
        knn_params,
        score,
        "scoring_roc_auc",
        assets,  # retriever name / scores / predictions
        module_dump_dir="",
    )
    labels, scores = get_prediction_evaluation_data(context)

    return scores, labels, context.data_handler.tags


def test_predict_returns_correct_indices(context, tmp_path):
    predictor = ThresholdPredictor(0.5)
    predictor.fit(*get_fit_data(context("multiclass"), tmp_path))
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))


def test_predict_returns_list(context, tmp_path):
    predictor = ThresholdPredictor(np.array([0.5, 0.5, 0.5]), n_classes=3)
    predictor.fit(*get_fit_data(context("multiclass"), tmp_path))
    scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))


def test_predict_handles_single_class(context, tmp_path):
    predictor = ThresholdPredictor(0.5)
    predictor.fit(*get_fit_data(context("multiclass"), tmp_path))
    scores = np.array([[0.5], [0.5], [0.5]])
    predictions = predictor.predict(scores)
    np.testing.assert_array_equal(predictions, np.array([0, 0, 0]))
