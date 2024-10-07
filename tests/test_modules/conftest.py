import pytest

from autointent import Context


@pytest.fixture
def context(load_clinic_subset):
    return Context(
        multiclass_intent_records=load_clinic_subset,
        multilabel_utterance_records=[],
        test_utterance_records=[],
        device="cpu",
        mode="multiclass",
        multilabel_generation_config="",
        db_dir="multiclass",
        regex_sampling=0,
        seed=0,
    )
