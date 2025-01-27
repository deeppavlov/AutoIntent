import importlib.resources as ires

import pytest

from autointent import Dataset, Ranker
from autointent.context.data_handler import DataHandler


@pytest.fixture
def data_handler():
    data_path = ires.files("tests.assets.data").joinpath("clinc_subset.json")
    return DataHandler(dataset=Dataset.from_json(data_path), random_seed=42)


def test_nli_transformer_predict_without_trained_head(data_handler):
    model = Ranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", train_classifier=True)
    with pytest.raises(ValueError, match="Classifier is not trained yet"):
        model.predict(data_handler.train_utterances(0))


def build_pairs(texts):
    return [[texts[0], t] for t in texts[1:]]


def check_predictions(predicted, labels):
    # we compare first text with others, hence we expect that minimum prob in the class is bigger max prob out the class
    min_in_class = float("inf")
    max_out_class = float("-inf")
    ref_label = labels[0]
    for prob, lbl in zip(predicted, labels[1:], strict=False):
        if ref_label == lbl:
            min_in_class = min(min_in_class, prob)
        else:
            max_out_class = max(max_out_class, prob)
    assert min_in_class > max_out_class


def check_ranking(ranked, labels):
    for r in ranked:
        assert "corpus_id" in r
        assert "score" in r

    # We expect the same label text to go first, then the rest
    ranked_labels = [labels[r["corpus_id"] + 1] == labels[0] for r in ranked]
    expected_labels = [labels[0] == lbl for lbl in labels[1:]]
    expected_labels.sort(reverse=True)
    assert ranked_labels == expected_labels


def test_nli_transformer_predict_with_train_head(data_handler):
    model = Ranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", train_classifier=True)
    texts = data_handler.train_utterances(0)
    labels = data_handler.train_labels(0)
    model.fit(texts, labels)
    predicted = model.predict(build_pairs(texts))
    check_predictions(predicted, labels)

    ranked = model.rank(texts[0], texts[1:])
    check_ranking(ranked, labels)


def test_nli_transformer_predict_default(data_handler):
    model = Ranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    texts = data_handler.train_utterances(0)
    labels = data_handler.train_labels(0)
    predicted = model.predict(build_pairs(texts))
    check_predictions(predicted, labels)

    ranked = model.rank(texts[0], texts[1:])
    check_ranking(ranked, labels)


def test_nli_transformer_predict_default_with_fit(data_handler):
    model = Ranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    texts = data_handler.train_utterances(0)
    labels = data_handler.train_labels(0)
    model.fit(texts, labels)
    predicted = model.predict(build_pairs(texts))
    check_predictions(predicted, labels)

    ranked = model.rank(texts[0], texts[1:])
    check_ranking(ranked, labels)
