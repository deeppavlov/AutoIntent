import numpy as np

from autointent import Context
from autointent.metrics import retrieval_hit_rate_macro, scoring_f1
from autointent.modules import VectorDBModule
from autointent.modules.scoring.mlknn.mlknn import MLKnnScorer


def test_base_mlknn(setup_environment, load_clinic_subset):
    run_name, db_dir = setup_environment

    utterance = [
        {
            "utterance": "why is there a hold on my american saving bank account",
            "labels": [0, 1, 2],
        },
        {
            "utterance": "i am nost sure why my account is blocked",
            "labels": [0, 2],
        },
    ]
    context = Context(
        multiclass_intent_records=load_clinic_subset,
        multilabel_utterance_records=utterance,
        test_utterance_records=utterance,
        device="cpu",
        mode="multiclass_as_multilabel",
        multilabel_generation_config="",
        db_dir=db_dir,
        regex_sampling=0,
        seed=0,
    )

    retrieval_params = {"k": 3, "model_name": "sergeyzh/rubert-tiny-turbo"}
    vector_db = VectorDBModule(**retrieval_params)
    vector_db.fit(context)
    metric_value, _ = vector_db.score(context, retrieval_hit_rate_macro)
    artifact = vector_db.get_assets()
    context.optimization_info.log_module_optimization(
        node_type="retrieval",
        module_type="vector_db",
        module_params=retrieval_params,
        metric_value=metric_value,
        metric_name="retrieval_hit_rate_macro",
        artifact=artifact,
    )

    scorer = MLKnnScorer(k=3)
    scorer.fit(context)
    score, predictions = scorer.score(context, scoring_f1)
    np.testing.assert_almost_equal(score, 0.8632478632478633)

    predictions = scorer.predict_labels(
        [
            "why is there a hold on my american saving bank account",
            "i am nost sure why my account is blocked",
            "why is there a hold on my capital one checking account",
            "i think my account is blocked but i do not know the reason",
            "can you tell me why is my bank account frozen",
        ]
    )
    assert (predictions == np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])).all()
