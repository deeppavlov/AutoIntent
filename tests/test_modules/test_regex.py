from autointent import Context
from autointent.metrics import retrieval_hit_rate, scoring_roc_auc
from autointent.modules import RegExp, VectorDBModule


def test_base_regex(setup_environment):
    run_name, db_dir = setup_environment
    data = [
        {
            "intent_id": 0,
            "intent_name": "accept_reservations",
            "sample_utterances": [
                "can i make a reservation for redrobin",
                "is it possible to make a reservation at redrobin",
                "does redrobin take reservations",
                "are reservations taken at redrobin",
                "does redrobin do reservations",
            ],
            "regexp_full_match": [".*"],
            "regexp_partial_match": [".*"],
        },
        {
            "intent_id": 1,
            "intent_name": "account_blocked",
            "sample_utterances": [
                "why is there a hold on my american saving bank account",
                "i am nost sure why my account is blocked",
                "why is there a hold on my capital one checking account",
                "i think my account is blocked but i do not know the reason",
                "can you tell me why is my bank account frozen",
            ],
            "regexp_full_match": [".*"],
            "regexp_partial_match": [".*"],
        },
    ]
    context = Context(
        multiclass_intent_records=data,
        multilabel_utterance_records=[],
        test_utterance_records=[],
        device="cpu",
        mode="multiclass",
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
