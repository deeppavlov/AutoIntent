import numpy as np
import pytest

from autointent import Context
from autointent.metrics import retrieval_hit_rate, scoring_roc_auc
from autointent.modules import DNNCScorer, VectorDBModule


@pytest.mark.xfail(reason="This test is failing on windows, because have different score")
@pytest.mark.parametrize(("train_head", "pred_score"), [(True, 1), (False, 0.5)])
def test_base_dnnc(setup_environment, load_clinic_subset, train_head, pred_score):
    run_name, db_dir, dump_dir, logs_dir = setup_environment

    context = Context(
        dataset=load_clinic_subset,
        dump_dir=dump_dir,
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

    scorer = DNNCScorer("sergeyzh/rubert-tiny-turbo", k=3, train_head=train_head)

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
    np.testing.assert_almost_equal(np.array([[0.0, pred_score, 0.0]] * len(test_data)), predictions, decimal=2)
    scorer.clear_cache()
