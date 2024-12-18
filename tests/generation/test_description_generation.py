from collections import defaultdict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from autointent import Dataset
from autointent.generation.description_generation import (
    create_intent_description,
    enhance_dataset_with_descriptions,
    generate_intent_descriptions,
    group_utterances_by_label,
)
from autointent.generation.prompt_scheme import PromptDescription
from autointent.schemas import Intent, Sample


def test_get_utterances_by_id_empty_input():
    utterances = []
    result = group_utterances_by_label(utterances)
    assert result == {}


def test_get_utterances_by_id_single_multiclass_utterance():
    samples = [Sample(utterance="Hello", label=1)]
    result = group_utterances_by_label(samples)
    assert result == {1: ["Hello"]}


def test_get_utterances_by_id_multiple_multiclass_same_label():
    samples = [Sample(utterance="Hello", label=1), Sample(utterance="Hi", label=1)]
    result = group_utterances_by_label(samples)
    assert result == {1: ["Hello", "Hi"]}


def test_get_utterances_by_id_single_multilabel_utterance():
    samples = [Sample(utterance="Good morning", label=[1, 2])]
    result = group_utterances_by_label(samples)
    expected_result = {1: ["Good morning"], 2: ["Good morning"]}
    assert result == expected_result


def test_get_utterances_by_id_multiple_multilabel_utterances():
    samples = [Sample(utterance="Good morning", label=[1, 2]), Sample(utterance="Good night", label=[1, 3])]
    result = group_utterances_by_label(samples)
    expected_result = {1: ["Good morning", "Good night"], 2: ["Good morning"], 3: ["Good night"]}
    assert result == expected_result


def test_get_utterances_by_id_oos_utterances():
    samples = [Sample(utterance="Unknown command", label=None), Sample(utterance="Hello", label=[2])]
    result = group_utterances_by_label(samples)
    assert result == {2: ["Hello"]}


def test_get_utterances_by_id_mixed_types():
    samples = [
        Sample(utterance="Hello", label=1),
        Sample(utterance="Good morning", label=[1, 3]),
        Sample(utterance="Random text", label=None),
        Sample(utterance="Hi", label=3),
    ]
    result = group_utterances_by_label(samples)
    expected_result = {1: ["Hello", "Good morning"], 3: ["Good morning", "Hi"]}
    assert result == expected_result


def test_get_utterances_by_id_duplicate_texts_different_labels():
    samples = [Sample(utterance="Duplicate", label=1), Sample(utterance="Duplicate", label=2)]
    result = group_utterances_by_label(samples)
    expected_result = {1: ["Duplicate"], 2: ["Duplicate"]}
    assert result == expected_result


@pytest.mark.asyncio
async def test_create_intent_description_basic():
    client = AsyncMock()
    mock_create = client.chat.completions.create
    mock_create.return_value = AsyncMock(choices=[Mock(message=Mock(content="Generated description"))])

    utterances = ["Hi", "Hello"]
    regexp_patterns = ["^hello$", "^hi$"]
    prompt = PromptDescription(
        text="Describe intent {intent_name} with examples: {user_utterances} and patterns: {regexp_patterns}",
    )

    description = await create_intent_description(
        client=client,
        intent_name="Greeting",
        utterances=utterances,
        regexp_patterns=regexp_patterns,
        prompt=prompt,
        model_name="gpt4o-mini",
    )

    assert description == "Generated description"
    mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_create_intent_description_empty_intent_name():
    client = AsyncMock()
    mock_create = client.chat.completions.create
    mock_create.return_value = AsyncMock(choices=[Mock(message=Mock(content="Generated description"))])

    utterances = ["Hi", "Hello"]
    regexp_patterns = ["^hello$", "^hi$"]
    prompt = PromptDescription(
        text="Describe intent {intent_name} with examples: {user_utterances} and patterns: {regexp_patterns}",
    )

    description = await create_intent_description(
        client=client,
        intent_name=None,
        utterances=utterances,
        regexp_patterns=regexp_patterns,
        prompt=prompt,
        model_name="gpt4o-mini",
    )

    assert description == "Generated description"
    mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_create_intent_description_empty_utterances_patterns():
    client = AsyncMock()
    mock_create = client.chat.completions.create
    mock_create.return_value = AsyncMock(choices=[Mock(message=Mock(content="Generated description"))])

    utterances = []
    regexp_patterns = []
    prompt = PromptDescription(
        text="Describe intent {intent_name} with examples: {user_utterances} and patterns: {regexp_patterns}",
    )

    description = await create_intent_description(
        client=client,
        intent_name="Greeting",
        utterances=utterances,
        regexp_patterns=regexp_patterns,
        prompt=prompt,
        model_name="gpt4o-mini",
    )

    assert description == "Generated description"
    mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_create_intent_description_large_utterances_patterns():
    client = AsyncMock()
    mock_create = client.chat.completions.create
    mock_create.return_value = AsyncMock(choices=[Mock(message=Mock(content="Generated description"))])

    utterances = ["Hi", "Hello", "Hey", "Greetings", "Salutations", "Good day", "Hiya"]
    regexp_patterns = ["^hello$", "^hi$", "^hey$", "^greetings$"]
    prompt = PromptDescription(
        text="Describe intent {intent_name} with examples: {user_utterances} and patterns: {regexp_patterns}",
    )

    with patch("random.sample", side_effect=lambda x, k: x[:k]) as mock_sample:
        description = await create_intent_description(
            client=client,
            intent_name="Greeting",
            utterances=utterances,
            regexp_patterns=regexp_patterns,
            prompt=prompt,
            model_name="gpt4o-mini",
        )

    assert description == "Generated description"
    mock_create.assert_called_once()
    mock_sample.assert_any_call(utterances, 5)
    mock_sample.assert_any_call(regexp_patterns, 3)


@pytest.mark.asyncio
async def test_generate_intent_descriptions_basic():
    client = AsyncMock()
    mock_create = client.chat.completions.create
    mock_create.return_value = AsyncMock(choices=[Mock(message=Mock(content="Generated description"))])

    intent_utterances = {1: ["Hello", "Hi"], 2: ["Goodbye", "See you"]}
    intents = [
        Intent(id=1, name="Greeting", description=None, regexp_full_match=[], regexp_partial_match=[]),
        Intent(id=2, name="Farewell", description=None, regexp_full_match=[], regexp_partial_match=[]),
    ]
    prompt = PromptDescription(
        text="Describe intent {intent_name} with examples: {user_utterances} and patterns: {regexp_patterns}",
    )
    updated_intents = await generate_intent_descriptions(
        client=client,
        intent_utterances=intent_utterances,
        intents=intents,
        prompt=prompt,
        model_name="gpt4o-mini",
    )

    assert all(intent.description == "Generated description" for intent in updated_intents)
    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_generate_intent_descriptions_skip_existing_descriptions():
    client = AsyncMock()
    mock_create = client.chat.completions.create
    mock_create.return_value = AsyncMock(choices=[Mock(message=Mock(content="Generated description"))])

    intent_utterances = {1: ["Hello", "Hi"], 2: ["Goodbye", "See you"]}
    intents = [
        Intent(
            id=1,
            name="Greeting",
            description="Existing description",
            regexp_full_match=[],
            regexp_partial_match=[],
        ),
        Intent(id=2, name="Farewell", description=None, regexp_full_match=[], regexp_partial_match=[]),
    ]
    prompt = PromptDescription(
        text="Describe intent {intent_name} with examples: {user_utterances} and patterns: {regexp_patterns}",
    )
    updated_intents = await generate_intent_descriptions(
        client=client,
        intent_utterances=intent_utterances,
        intents=intents,
        prompt=prompt,
        model_name="gpt4o-mini",
    )

    assert updated_intents[0].description == "Existing description"
    assert updated_intents[1].description == "Generated description"
    assert mock_create.call_count == 1  # Only one call for the second intent


@pytest.mark.asyncio
async def test_generate_intent_descriptions_empty_utterances_patterns():
    client = AsyncMock()
    mock_create = client.chat.completions.create
    mock_create.return_value = AsyncMock(choices=[Mock(message=Mock(content="Generated description"))])

    intent_utterances = {}  # No utterances for any intent
    intents = [
        Intent(id=1, name="Greeting", description=None, regexp_full_match=[], regexp_partial_match=[]),
    ]
    prompt = PromptDescription(
        text="Describe intent {intent_name} with examples: {user_utterances} and patterns: {regexp_patterns}",
    )
    updated_intents = await generate_intent_descriptions(
        client=client,
        intent_utterances=intent_utterances,
        intents=intents,
        prompt=prompt,
        model_name="gpt4o-mini",
    )

    # Assertions
    assert updated_intents[0].description == "Generated description"
    mock_create.assert_called_once_with(
        messages=[
            {
                "role": "user",
                "content": prompt.text.format(intent_name="Greeting", user_utterances="", regexp_patterns=""),
            },
        ],
        model="gpt4o-mini",
        temperature=0.2,
    )


def test_enhance_dataset_with_descriptions_basic():
    client = AsyncMock()
    with patch(
        "autointent.generation.description_generation.generate_intent_descriptions",
        new=AsyncMock(
            return_value=[
                Intent(id=1, name="Greeting", description="Generated description"),
                Intent(id=2, name="Farewell", description="Generated description"),
            ],
        ),
    ) as mock_generate_intent_descriptions:
        dataset = Dataset.from_dict(
            {
                "train": [
                    {"utterance": "Hello", "label": 0},
                    {"utterance": "Goodbye", "label": 1},
                ],
                "intents": [
                    {"id": 0, "name": "Greeting"},
                    {"id": 1, "name": "Farewell"},
                ],
            },
        )
        prompt = PromptDescription(
            text="Describe intent {intent_name} with examples: {user_utterances} and patterns: {regexp_patterns}",
        )
        enhanced_dataset = enhance_dataset_with_descriptions(
            dataset=dataset,
            client=client,
            prompt=prompt,
            model_name="gpt4o-mini",
        )
        expected_intent_utterances = defaultdict(list, {0: ["Hello"], 1: ["Goodbye"]})

        assert enhanced_dataset.intents[0].description == "Generated description"
        assert enhanced_dataset.intents[1].description == "Generated description"
        mock_generate_intent_descriptions.assert_called_once_with(
            client,
            expected_intent_utterances,
            [
                Intent(id=0, name="Greeting", description=None, regexp_full_match=[], regexp_partial_match=[]),
                Intent(id=1, name="Farewell", description=None, regexp_full_match=[], regexp_partial_match=[]),
            ],
            prompt,
            "gpt4o-mini",
        )


def test_enhance_dataset_with_existing_descriptions():
    client = AsyncMock()
    with patch(
        "autointent.generation.description_generation.generate_intent_descriptions",
        new=AsyncMock(
            return_value=[
                Intent(id=0, name="Greeting", description="Existing description"),
                Intent(id=1, name="Farewell", description="Generated description"),
            ],
        ),
    ) as mock_generate_intent_descriptions:
        dataset = Dataset.from_dict(
            {
                "train": [
                    {"utterance": "Hello", "label": 0},
                    {"utterance": "Goodbye", "label": 1},
                ],
                "intents": [
                    {"id": 0, "name": "Greeting", "description": "Existing description"},
                    {"id": 1, "name": "Farewell"},
                ],
            },
        )
        prompt = PromptDescription(
            text="Describe intent {intent_name} with examples: {user_utterances} and patterns: {regexp_patterns}",
        )
        enhanced_dataset = enhance_dataset_with_descriptions(
            dataset=dataset,
            client=client,
            prompt=prompt,
            model_name="gpt4o-mini",
        )
        expected_intent_utterances = defaultdict(list, {0: ["Hello"], 1: ["Goodbye"]})

        assert enhanced_dataset.intents[0].description == "Existing description"
        assert enhanced_dataset.intents[1].description == "Generated description"
        mock_generate_intent_descriptions.assert_called_once_with(
            client,
            expected_intent_utterances,
            [
                Intent(
                    id=0,
                    name="Greeting",
                    description="Existing description",
                    regexp_full_match=[],
                    regexp_partial_match=[],
                ),
                Intent(id=1, name="Farewell", description=None, regexp_full_match=[], regexp_partial_match=[]),
            ],
            prompt,
            "gpt4o-mini",
        )
