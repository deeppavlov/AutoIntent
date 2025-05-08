import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from autointent import Embedder, Ranker, VectorIndex
from autointent._dump_tools import Dumper
from autointent.configs import CrossEncoderConfig, EmbedderConfig, TokenizerConfig
from autointent.schemas import Tag, TagsList


class TestSimpleAttributes:
    def init_attributes(self):
        self.integer = 1
        self.float = 1.0
        self.string = "test"
        self.boolean = True
        self.array = np.array([1, 2, 3])

    def check_attributes(self):
        assert self.integer == 1
        assert self.float == 1.0
        assert self.string == "test"
        assert self.boolean
        np.testing.assert_array_equal(self.array, np.array([1, 2, 3]))


class TestTags:
    def init_attributes(self):
        self.tags = TagsList(tags=[Tag(name="hello", intent_ids=[0]), Tag(name="world", intent_ids=[1])])

    def check_attributes(self):
        assert self.tags == TagsList(tags=[Tag(name="hello", intent_ids=[0]), Tag(name="world", intent_ids=[1])])


class TestTransformers:
    def init_attributes(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self._tokenizer_predictions = np.array(self.tokenizer(["hello", "world"]).input_ids)
        self.transformer = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

        with torch.no_grad():
            self._transformer_predictions = (
                self.transformer(input_ids=torch.tensor(self._tokenizer_predictions)).logits.cpu().numpy()
            )

    def check_attributes(self):
        tokenizer_predictions = self.tokenizer(["hello", "world"]).input_ids
        np.testing.assert_array_equal(self._tokenizer_predictions, tokenizer_predictions)
        with torch.no_grad():
            np.testing.assert_array_equal(
                self._transformer_predictions,
                self.transformer(input_ids=torch.tensor(tokenizer_predictions)).logits.cpu().numpy(),
            )


class TestVectorIndex:
    def init_attributes(self):
        self.vector_index = VectorIndex(
            embedder_config=EmbedderConfig.from_search_config("bert-base-uncased"),
        )
        self.vector_index.add(texts=["hello", "world"], labels=[0, 1])

    def check_attributes(self):
        assert self.vector_index.texts == ["hello", "world"]
        assert self.vector_index.labels == [0, 1]


class TestEmbedder:
    def init_attributes(self):
        self.embedder = Embedder(
            embedder_config=EmbedderConfig.from_search_config("bert-base-uncased"),
        )
        self._embedder_predictions = self.embedder.embed(["hello", "world"])

    def check_attributes(self):
        np.testing.assert_array_equal(
            self._embedder_predictions,
            self.embedder.embed(["hello", "world"]),
        )


class TestSklearnEstimator:
    def init_attributes(self):
        self.estimator = LogisticRegression()
        self.estimator.fit([[1, 2, 3], [4, 5, 6]], [0, 1])
        self._estimator_predictions = self.estimator.predict([[1, 2, 3], [4, 5, 6]])

    def check_attributes(self):
        np.testing.assert_array_equal(
            self._estimator_predictions,
            self.estimator.predict([[1, 2, 3], [4, 5, 6]]),
        )


class TestRanker:
    def init_attributes(self):
        self.ranker = Ranker(
            cross_encoder_config={"model_name": "cross-encoder/ms-marco-MiniLM-L6-v2", "train_head": True},
        )
        self.ranker.fit(
            ["hello", "world", "bye", "earth", "hello", "world", "bye", "earth"],
            [0, 1, 0, 1, 0, 1, 0, 1],
        )
        self._ranker_predictions = self.ranker.predict(
            [("hello", "world"), ("bye", "earth")],
        )

    def check_attributes(self):
        np.testing.assert_array_equal(
            self._ranker_predictions,
            self.ranker.predict([("hello", "world"), ("bye", "earth")]),
        )


class TestEmbedderConfig:
    def init_attributes(self):
        self.pydantic_model = EmbedderConfig(
            model_name="bert-base-uncased",
            batch_size=16,
            device="cpu",
            trust_remote_code=True,
            tokenizer_config=TokenizerConfig(max_length=512, padding="longest", truncation=False),
        )

    def check_attributes(self):
        assert self.pydantic_model.model_name == "bert-base-uncased"
        assert self.pydantic_model.batch_size == 16
        assert self.pydantic_model.device == "cpu"
        assert self.pydantic_model.trust_remote_code
        assert self.pydantic_model.tokenizer_config.max_length == 512
        assert self.pydantic_model.tokenizer_config.padding == "longest"
        assert not self.pydantic_model.tokenizer_config.truncation


class TestCrossEncoderConfig:
    def init_attributes(self):
        self.pydantic_model = CrossEncoderConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            train_head=True,
            device="cpu",
            batch_size=16,
            trust_remote_code=True,
            tokenizer_config=TokenizerConfig(max_length=512, padding="longest", truncation=False),
        )

    def check_attributes(self):
        assert self.pydantic_model.model_name == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert self.pydantic_model.train_head
        assert self.pydantic_model.device == "cpu"
        assert self.pydantic_model.batch_size == 16
        assert self.pydantic_model.trust_remote_code
        assert self.pydantic_model.tokenizer_config.max_length == 512
        assert self.pydantic_model.tokenizer_config.padding == "longest"
        assert not self.pydantic_model.tokenizer_config.truncation


@pytest.mark.parametrize(
    "test_class",
    [
        TestSimpleAttributes,
        TestTags,
        TestTransformers,
        TestVectorIndex,
        TestEmbedder,
        TestSklearnEstimator,
        TestRanker,
        TestEmbedderConfig,
        TestCrossEncoderConfig,
    ],
)
def test_dumper(test_class):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        test_obj = test_class()
        test_obj.init_attributes()

        Dumper.dump(test_obj, Path(temp_dir))
        del test_obj

        loaded_obj = test_class()
        Dumper.load(loaded_obj, Path(temp_dir))
        loaded_obj.check_attributes()
