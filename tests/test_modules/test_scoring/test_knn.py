import numpy as np

from autointent import Context
from autointent.metrics import retrieval_hit_rate, scoring_roc_auc
from autointent.modules import KNNScorer, VectorDBModule


def test_base_knn(setup_environment, load_clinic_subset):
    run_name, db_dir = setup_environment

    context = Context(
        multiclass_intent_records=load_clinic_subset,
        multilabel_utterance_records=[],
        test_utterance_records=[],
        device="cpu",
        mode="multiclass",
        multilabel_generation_config="",
        regex_sampling=0,
        seed=0,
        db_dir=db_dir,
    )

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
    )

    scorer = KNNScorer(k=3, weights="distance")

    scorer.fit(context)
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
