import numpy as np

from autointent import Context
from autointent.context.data_handler import Dataset
from autointent.metrics import retrieval_hit_rate_macro, scoring_f1
from autointent.modules import VectorDBModule
from autointent.modules.scoring.mlknn.mlknn import MLKnnScorer
from tests.conftest import load_clinc_subset


def test_base_mlknn(setup_environment):
    db_dir, dump_dir, logs_dir = setup_environment

    dataset = load_clinc_subset("multilabel")

    test_dataset = Dataset.model_validate(
        {
            "utterances": [
                {
                    "text": "why is there a hold on my american saving bank account",
                    "label": [0, 1, 2],
                },
                {
                    "text": "i am nost sure why my account is blocked",
                    "label": [0, 2],
                },
            ],
        },
    )

    context = Context(
        dataset=dataset,
        test_dataset=test_dataset,
        db_dir=db_dir(),
        dump_dir=dump_dir,
    )

    retrieval_params = {"k": 3, "model_name": "sergeyzh/rubert-tiny-turbo", "db_dir": db_dir()}
    vector_db = VectorDBModule(**retrieval_params)
    vector_db.fit(context.data_handler.utterances_train, context.data_handler.labels_train)
    metric_value = vector_db.score(context, retrieval_hit_rate_macro)
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

    scorer = MLKnnScorer(db_dir=db_dir(), k=3, model_name="sergeyzh/rubert-tiny-turbo", n_classes=3)
    scorer.fit(context.data_handler.utterances_train, context.data_handler.labels_train)
    score = scorer.score(context, scoring_f1)
    np.testing.assert_almost_equal(score, 2 / 9)

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
