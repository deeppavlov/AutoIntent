import numpy as np

from autointent import Context
from autointent.metrics import scoring_roc_auc
from autointent.modules import VectorDBModule
from autointent.modules.scoring.mlknn.mlknn import MLKnnScorer
from autointent.pipeline.main import get_db_dir, get_run_name, load_data, setup_logging


def test_base_mlknn():
    setup_logging("DEBUG")
    run_name = get_run_name("multiclass-cpu")
    db_dir = get_db_dir("", run_name)

    data = load_data("tests/minimal-optimization/data/clinc_subset.json", multilabel=False)
    utterance = [
        {
            "utterance": "why is there a hold on my american saving bank account",
            "labels": [1],
        },
        {
            "utterance": "i am nost sure why my account is blocked",
            "labels": [0],
        },
    ]
    context = Context(
        multiclass_intent_records=data,
        multilabel_utterance_records=utterance,
        test_utterance_records=utterance,
        device="cpu",
        mode="multilabel",
        multilabel_generation_config="",
        db_dir=db_dir,
        regex_sampling=0,
        seed=0,
    )

    vector_db = VectorDBModule(k=3, model_name="sergeyzh/rubert-tiny-turbo")
    vector_db.fit(context)
    scorer = MLKnnScorer(k=3)

    context.optimization_logs.cache["best_assets"]["retrieval"] = "sergeyzh/rubert-tiny-turbo"
    scorer.fit(context)
    assert scorer.score(context, scoring_roc_auc) == 0.5
    predictions = scorer.predict(
        np.array(
            [
                "why is there a hold on my american saving bank account",
                "i am nost sure why my account is blocked",
                "why is there a hold on my capital one checking account",
                "i think my account is blocked but i do not know the reason",
                "can you tell me why is my bank account frozen",
            ]
        )
    )
    assert (
        predictions == np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    ).all()
