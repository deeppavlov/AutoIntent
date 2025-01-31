from unittest.mock import Mock

from autointent.generation.utterances.basic.chat_template import SynthesizerChatTemplate


def has_unfilled_fields(template):
    try:
        # Attempt to format the string with empty values
        template.format(**{})  # noqa: PIE804
        return False  # No unfilled fields  # noqa: TRY300
    except KeyError:
        return True  # Unfilled fields detected


def test_default_chat_template(dataset):
    template = SynthesizerChatTemplate(dataset, split="train_0")
    prompt = template(dataset.intents[0], n_examples=1)
    for msg in prompt:
        assert not has_unfilled_fields(msg["content"])
    assert "extra_instructions" not in prompt


def test_extra_instructions(dataset):
    template = SynthesizerChatTemplate(dataset, split="train_0", extra_instructions="football")
    prompt = template(dataset.intents[0], n_examples=1)[0]["content"]
    assert "extra_instructions" not in prompt
    assert "football" in prompt


