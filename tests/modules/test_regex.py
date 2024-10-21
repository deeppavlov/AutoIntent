import pytest

from autointent import Context
from autointent.context.data_handler import Dataset
from autointent.metrics import retrieval_hit_rate, scoring_roc_auc
from autointent.modules import RegExp, VectorDBModule


@pytest.mark.xfail(reason="Issues with intent_id")
def test_base_regex(setup_environment):
    run_name, db_dir = setup_environment

    data = {
        "utterances": [
            {
                "text": "can i make a reservation for redrobin",
                "label": 0,
            },
            {
                "text": "is it possible to make a reservation at redrobin",
                "label": 0,
            },
            {
                "text": "does redrobin take reservations",
                "label": 0,
            },
            {
                "text": "are reservations taken at redrobin",
                "label": 0,
            },
            {
                "text": "does redrobin do reservations",
                "label": 0,
            },
            {
                "text": "why is there a hold on my american saving bank account",
                "label": 1,
            },
            {
                "text": "i am nost sure why my account is blocked",
                "label": 1,
            },
            {
                "text": "why is there a hold on my capital one checking account",
                "label": 1,
            },
            {
                "text": "i think my account is blocked but i do not know the reason",
                "label": 1,
            },
            {
                "text": "can you tell me why is my bank account frozen",
                "label": 1,
            },
        ],
        "intents": [
            {
                "id": 0,
                "name": "accept_reservations",
                "regexp_full_match": [".*"],
                "regexp_partial_match": [".*"],
            },
            {
                "id": 1,
                "name": "account_blocked",
                "regexp_full_match": [".*"],
                "regexp_partial_match": [".*"],
            },
        ],
    }

    context = Context(
        dataset=Dataset.model_validate(data),
        test_dataset=None,
        device="cpu",
        multilabel_generation_config="",
        regex_sampling=0,
        seed=0,
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
        module_dump_dir="",
    )

    scorer = RegExp()

    scorer.fit(context)
    score, _ = scorer.score(context, scoring_roc_auc)
    assert score == 0.5
    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]
    predictions = scorer.predict(test_data)
    assert predictions == [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
