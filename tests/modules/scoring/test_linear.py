import numpy as np

from autointent.metrics import retrieval_hit_rate, scoring_roc_auc
from autointent.modules import LinearScorer, VectorDBModule


def test_base_linear(context, setup_environment):
    get_db_dir, dump_dir, logs_dir = setup_environment

    context = context("multiclass")

    retrieval_params = {"k": 3, "model_name": "sergeyzh/rubert-tiny-turbo", "db_dir": get_db_dir()}
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

    scorer = LinearScorer("sergeyzh/rubert-tiny-turbo")

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
                [0.17929172, 0.59130114, 0.22940714],
                [0.15927979, 0.62363961, 0.2170806],
                [
                    0.20069508,
                    0.53883687,
                    0.26046804,
                ],
                [0.17557001, 0.61310582, 0.21132417],
                [
                    0.17911179,
                    0.63123131,
                    0.1896569,
                ],
            ]
        ),
        predictions,
        decimal=2,
    )
