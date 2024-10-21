import numpy as np

from autointent.metrics import retrieval_hit_rate, scoring_roc_auc
from autointent.modules import LinearScorer, VectorDBModule


def test_base_linear(context):
    context = context("multiclass")

    retrieval_params = {"k": 3, "model_name": "sergeyzh/rubert-tiny-turbo"}
    vector_db = VectorDBModule(**retrieval_params)
    vector_db.fit(context)
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

    scorer = LinearScorer()

    scorer.fit(context)
    score = scorer.score(context, scoring_roc_auc)
    assert score == 1
    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]
    predictions = scorer.predict(test_data)

    np.testing.assert_almost_equal(
        np.array(
            [
                [0.17928316, 0.59134606, 0.22937078],
                [0.15927759, 0.62366319, 0.21705922],
                [0.20068982, 0.53887681, 0.26043337],
                [0.17557128, 0.61313277, 0.21129594],
                [0.17908815, 0.63129863, 0.18961322],
            ]
        ),
        predictions,
        decimal=2,
    )
