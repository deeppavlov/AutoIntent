from autointent import Context
from autointent.pipeline import Pipeline
from autointent.pipeline.main import get_db_dir, get_run_name, load_data, setup_logging


def test_multiclass():
    setup_logging("DEBUG")

    # configure the run and data
    run_name = get_run_name("multiclass-cpu")
    db_dir = get_db_dir("", run_name)

    # create shared objects for a whole pipeline
    data = load_data("tests/minimal-optimization/data/clinc_subset.json", multilabel=False)
    context = Context(
        multiclass_intent_records=data,
        multilabel_utterance_records=[],
        test_utterance_records=[],
        device="cpu",
        mode="multiclass",
        multilabel_generation_config="",
        db_dir=db_dir,
        regex_sampling=0,
        seed=0,
    )

    # run optimization
    pipeline = Pipeline(
        config_path="tests/minimal-optimization/configs/multiclass.yaml",
        mode="multiclass",
    )
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir="tests/minimal-optimization/logs", run_name=run_name)


def test_multilabel():
    setup_logging("DEBUG")

    # configure the run and data
    run_name = get_run_name("multilabel-cpu")
    db_dir = get_db_dir("", run_name)

    # create shared objects for a whole pipeline
    data = load_data("tests/minimal-optimization/data/clinc_subset.json", multilabel=False)
    context = Context(
        multiclass_intent_records=data,
        multilabel_utterance_records=[],
        test_utterance_records=[],
        device="cpu",
        mode="multiclass_as_multilabel",
        multilabel_generation_config="",
        db_dir=db_dir,
        regex_sampling=0,
        seed=0,
    )

    # run optimization
    pipeline = Pipeline(
        config_path="tests/minimal-optimization/configs/multilabel.yaml",
        mode="multiclass_as_multilabel",
    )
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir="tests/minimal-optimization/logs", run_name=run_name)
