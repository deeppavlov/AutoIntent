import numpy as np

from autointent.metrics import retrieval_hit_rate, scoring_roc_auc
from autointent.modules import LinearScorer, VectorDBModule


def test_base_linear(context, setup_environment):
    run_name, db_dir = setup_environment

    context = context("multiclass")

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

    scorer = LinearScorer(db_dir)

    scorer.fit(context.data_handler.utterances_train, context.data_handler.labels_train)
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
                [
                    0.10645702,
                    0.78091989,
                    0.11262309,
                ],
                [
                    0.06595671,
                    0.84008038,
                    0.09396291,
                ],
                [
                    0.12650829,
                    0.73910616,
                    0.13438556,
                ],
                [
                    0.06949465,
                    0.84499476,
                    0.08551059,
                ],
                [
                    0.07943653,
                    0.81020573,
                    0.11035774,
                ],
            ]
        ),
        predictions,
        decimal=2,
    )
