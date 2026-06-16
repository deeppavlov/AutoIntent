import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation import Generator
from autointent.generation.utterances import DatasetBalancer

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_generator() -> Mock:
    generator = Mock(spec=Generator)
    generator.get_chat_completion.return_value = "test_utterance"
    generator.get_chat_completion_async = AsyncMock(return_value="test_utterance")
    return generator


@pytest.fixture
def mock_prompt_maker() -> Mock:
    return Mock(return_value=[Mock()])


@pytest.fixture
def unbalanced_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "intents": [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}],
            "train": [
                {"utterance": "test a1", "label": 0},
                {"utterance": "test a2", "label": 0},
                {"utterance": "test b1", "label": 1},
            ],
        }
    )


def test_balancer(unbalanced_dataset: Dataset, mock_generator: Mock, mock_prompt_maker: Mock) -> None:
    balancer = DatasetBalancer(generator=mock_generator, prompt_maker=mock_prompt_maker)
    logger.info("Before balancing:")
    for sample in unbalanced_dataset[Split.TRAIN]:
        logger.info("Utterance: %s, Label: %s", sample["utterance"], sample["label"])

    with patch.object(balancer.utterance_generator, "__call__") as mock_call:
        mock_call.return_value = ["generated_utterance"]

        balanced = balancer.balance(unbalanced_dataset)

    logger.info("After balancing:")
    for sample in balanced[Split.TRAIN]:
        logger.info("Utterance: %s, Label: %s", sample["utterance"], sample["label"])

    labels = [s["label"] for s in balanced[Split.TRAIN]]
    assert labels.count(0) == 2, "Class 0 should not change"
    assert labels.count(1) == 2, "Class 1 should increase to 2"
    assert len(labels) == 4, "The total number of examples should be 4"

    original_utterances = {s["utterance"] for s in unbalanced_dataset[Split.TRAIN]}
    balanced_utterances = {s["utterance"] for s in balanced[Split.TRAIN]}
    assert original_utterances.issubset(balanced_utterances)
