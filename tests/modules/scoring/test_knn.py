import numpy as np

from autointent import Context
from autointent.metrics import retrieval_hit_rate, scoring_roc_auc
from autointent.modules import KNNScorer, VectorDBModule


def test_base_knn(setup_environment, load_clinc_subset):
    db_dir, dump_dir, logs_dir = setup_environment

    dataset = load_clinc_subset("multiclass")

    context = Context(
        dataset=dataset,
        dump_dir=dump_dir,
        db_dir=db_dir(),
    )

    retrieval_params = {"k": 3, "model_name": "sergeyzh/rubert-tiny-turbo", "db_dir": db_dir()}
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

    scorer = KNNScorer(
        k=3, weights="distance", model_name="sergeyzh/rubert-tiny-turbo", db_dir=db_dir(), n_classes=3, multilabel=False
    )

    scorer.fit(context.data_handler.utterances_train, context.data_handler.labels_train)
    score = scorer.score(context, scoring_roc_auc)
    assert score == 1
    predictions = scorer.predict(
        [
            "why is there a hold on my american saving bank account",
            "i am nost sure why my account is blocked",
            "why is there a hold on my capital one checking account",
            "i think my account is blocked but i do not know the reason",
            "can you tell me why is my bank account frozen",
        ]
    )
    assert (
        predictions == np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    ).all()
