import os
from collections import defaultdict
from unittest.mock import AsyncMock, Mock, patch

import pytest
from datasets import Dataset as HFDataset

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation.utterances import DatasetBalancer, Generator
from autointent.generation.utterances.basic.chat_template import SynthesizerChatTemplate
from autointent.schemas import Sample


@pytest.fixture
def mock_generator():
    generator = Mock(spec=Generator)
    generator.get_chat_completion.return_value = "test_utterance"
    generator.get_chat_completion_async = AsyncMock(return_value="test_utterance")
    return generator


@pytest.fixture
def mock_prompt_maker():
    return Mock(return_value=[Mock()])


@pytest.fixture
def unbalanced_dataset():
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


def test_balancer(unbalanced_dataset, mock_generator, mock_prompt_maker):
    balancer = DatasetBalancer(generator=mock_generator, prompt_maker=mock_prompt_maker)
    print("\nBefore balancing:")
    for sample in unbalanced_dataset[Split.TRAIN]:
        print(f"Utterance: {sample['utterance']}, Label: {sample['label']}")

    with patch.object(balancer.evolver, "augment") as mock_augment:

        def augment_side_effect(dataset, split_name, n_generations, update_split, batch_size):
            new_sample = {"utterance": "generated_utterance", "label": 1}
            if update_split:
                current_data = dataset[split_name].to_list()
                current_data.append(new_sample)
                dataset[split_name] = HFDataset.from_list(current_data)
            return [Sample(**new_sample)]

        mock_augment.side_effect = augment_side_effect

        balanced = balancer.balance(unbalanced_dataset)

    print("\nAfter balancing:")
    for sample in balanced[Split.TRAIN]:
        print(f"Utterance: {sample['utterance']}, Label: {sample['label']}")

    labels = [s["label"] for s in balanced[Split.TRAIN]]
    assert labels.count(0) == 2, "Class 0 should not change"
    assert labels.count(1) == 2, "Class 1 should increase to 2"
    assert len(labels) == 4, "The total number of examples should be 4"

    original_utterances = {s["utterance"] for s in unbalanced_dataset[Split.TRAIN]}
    balanced_utterances = {s["utterance"] for s in balanced[Split.TRAIN]}
    assert original_utterances.issubset(balanced_utterances)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key in environment")
def test_real_balancer():
    test_data = {
        "intents": [{"id": 0, "name": "Book restaurant"}, {"id": 1, "name": "Check weather"}],
        "train": [
            {"utterance": "Book a table for two", "label": 0},
            {"utterance": "Reserve a table", "label": 0},
            {"utterance": "What's the weather in Moscow?", "label": 1},
        ],
    }
    dataset = Dataset.from_dict(test_data)
    template = SynthesizerChatTemplate(dataset, split="train")
    generator = Generator()
    evolutions = template
    balancer = DatasetBalancer(generator=generator, prompt_maker=evolutions, max_samples_per_class=3, async_mode=False)

    print("\nStarting balance process...")
    balanced = balancer.balance(dataset)

    class_counts = defaultdict(int)
    for sample in balanced[Split.TRAIN]:
        class_counts[sample["label"]] += 1

    print("\nBalancing results:")
    print(f"Class 0 count: {class_counts[0]}")
    print(f"Class 1 count: {class_counts[1]}")
    print("\nGenerated examples:")
    for sample in balanced[Split.TRAIN]:
        if sample["utterance"] not in {s["utterance"] for s in test_data["train"]}:
            print(f"[Class {sample['label']}]: {sample['utterance']}")

    assert class_counts[0] == 3, "Class 0 should have 3 examples"
    assert class_counts[1] == 3, "Class 1 should have 3 examples"
    assert len(balanced[Split.TRAIN]) == 6, "Total examples should be 6"
