"""Prompt description configuration."""

from pydantic import BaseModel, field_validator

from autointent.generation.prompts import PROMPT_DESCRIPTION


class PromptDescription(BaseModel):
    """Prompt description configuration."""

    text: str = PROMPT_DESCRIPTION
    """
    The template for the prompt to generate descriptions for intents.
    Should include placeholders for {intent_name} and {user_utterances}.
    - `{intent_name}` will be replaced with the name of the intent.
    - `{user_utterances}` will be replaced with the user utterances related to the intent.
    - (optionally) `{regexp_patterns}` will be replaced with the regular expressions that match user utterances.
    """

    @classmethod
    @field_validator("text")
    def check_valid_prompt(cls, value: str) -> str:
        """
        Validate the prompt description template.

        :param value: Check the prompt description template.
        :return:
        """
        if value.find("{intent_name}") == -1 or value.find("{user_utterances}") == -1:
            text_error = (
                "The 'prompt_description' template must properly "
                "include {intent_name} and {user_utterances} placeholders."
            )
            raise ValueError(text_error)
        return value
