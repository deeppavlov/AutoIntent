import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from autointent.configs import SentenceTransformerEmbeddingConfig
from autointent import Pipeline
from autointent.context.data_handler import DataHandler
from autointent.modules import CatBoostScorer
from tests.conftest import get_test_embedder_config

_embedder_config = SentenceTransformerEmbeddingConfig(model_name="prajjwal1/bert-tiny", revision="refs/pr/16")

pytest.importorskip("catboost")


def test_catboost_scorer_dump_load(dataset):
    """Test that CatBoostScorer can be saved and loaded while preserving predictions."""
    data_handler = DataHandler(dataset)

    scorer_original = CatBoostScorer(
        embedder_config=get_test_embedder_config(),
        iterations=50,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="Accuracy",
        random_seed=42,
        verbose=False,
    )

    scorer_original.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my account",
        "why is my bank account frozen",
    ]

    predictions_before = scorer_original.predict(test_data)

    temp_dir_path = Path(tempfile.mkdtemp(prefix="catboost_scorer_test_"))
    try:
        scorer_original.dump(str(temp_dir_path))
        scorer_loaded = CatBoostScorer.load(str(temp_dir_path))

        assert hasattr(scorer_loaded, "_model")
        assert scorer_loaded._model is not None

        predictions_after = scorer_loaded.predict(test_data)
        assert predictions_before.shape == predictions_after.shape
        np.testing.assert_allclose(predictions_before, predictions_after, atol=1e-6)

    finally:
        shutil.rmtree(temp_dir_path, ignore_errors=True)  # workaround for windows permission error


def test_catboost_prediction_multilabel(dataset):
    """Test that the transformer model can fit and make predictions."""
    data_handler = DataHandler(dataset.to_multilabel())

    scorer = CatBoostScorer(
        embedder_config=_embedder_config,
        iterations=50,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="Accuracy",
        random_seed=42,
        verbose=False,
        val_fraction=None,
    )

    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]

    predictions = scorer.predict(test_data)
    assert np.allclose(
        predictions,
        np.array(
            [
                [0.37150982, 0.5935175, 0.36279131, 0.37357718],
                [0.37309364, 0.53746911, 0.38326219, 0.39884488],
                [0.37744044, 0.56529594, 0.37456834, 0.38646843],
                [0.41484185, 0.48539558, 0.41669755, 0.42929345],
                [0.38344306, 0.58516115, 0.37940454, 0.39640789],
            ]
        ),
        rtol=0.01,
    )


@pytest.mark.parametrize("features_type", ["text", "embedding", "both"])
@pytest.mark.parametrize("use_embedding_features", [True, False])
def test_catboost_features_types(dataset, features_type, use_embedding_features):
    """Test that CatBoostScorer works properly without an embedder (using BoW encoding)."""
    data_handler = DataHandler(dataset)

    scorer = CatBoostScorer(
        embedder_config=_embedder_config,
        iterations=50,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="Accuracy",
        random_seed=42,
        features_type=features_type,
        use_embedding_features=use_embedding_features,
        verbose=False,
    )

    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))

    test_data = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]

    predictions = scorer.predict(test_data)
    assert predictions.shape[0] == len(test_data)
    assert predictions.shape[1] == len(set(data_handler.train_labels(0)))
    assert 0.0 <= np.min(predictions) <= np.max(predictions) <= 1.0


def test_catboost_cache_clearing(dataset):
    """Test that the transformer model properly handles cache clearing."""
    data_handler = DataHandler(dataset)
    scorer = CatBoostScorer(
        embedder_config=get_test_embedder_config(),
        iterations=50,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="Accuracy",
        random_seed=42,
        verbose=False,
    )
    scorer.fit(data_handler.train_utterances(0), data_handler.train_labels(0))
    test_data = ["test text"]
    scorer.predict(test_data)
    scorer.clear_cache()
    with pytest.raises(RuntimeError):
        scorer.predict(test_data)


def test_catboost_in_pipeline(dataset):
    """Test CatBoostScorer as part of an AutoML pipeline."""
    search_space = [
        {
            "node_type": "scoring",
            "target_metric": "scoring_roc_auc",
            "search_space": [
                {
                    "module_name": "catboost",
                    "iterations": [50],
                    "learning_rate": [0.05],
                    "features_type": ["embedding"],
                }
            ],
        },
        {"node_type": "decision", "target_metric": "decision_accuracy", "search_space": [{"module_name": "argmax"}]},
    ]

    pipeline = Pipeline.from_search_space(search_space)
    pipeline.set_config(get_test_embedder_config())
    pipeline.fit(dataset)
    predictions = pipeline.predict(["test utterance"])
    assert len(predictions) == 1
