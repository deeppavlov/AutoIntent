"""Prompt description configuration."""

from pydantic import BaseModel, field_validator

PROMPT_DESCRIPTION = """
Your task is to write a description of the intent.

You are given the name of the intent, user intentions related to it, and
regular expressions that match user utterances. The description should be:
1) In declarative form.
2) No more than one sentence.
3) In the language in which the utterances or regular expressions are written.

Remember:
- Respond with just the description, no extra details.
- Keep in mind that either the names, user queries, or regex patterns may not be provided.

For example:

name:
activate_my_card
user utterances:
Please help me with my card. It won't activate.
I tried but am unable to activate my card.
I want to start using my card.
regex patterns:
(activate.*card)|(start.*using.*card)
description:
User wants to activate his card.

name:
beneficiary_not_allowed
user utterances:

regex patterns:
(not.*allowed.*beneficiary)|(cannot.*add.*beneficiary)
description:
User wants to know why his beneficiary is not allowed.

name:
vacation_registration
user utterances:
как оформить отпуск
в какие даты надо оформить отпуск
как запланировать отпуск
regex patterns:

description:
Пользователь спрашивает про оформление отпуска.

name:
{intent_name}
user utterances:
{user_utterances}
regex patterns:
{regex_patterns}
description:

"""


class PromptDescription(BaseModel):
    """Prompt description configuration."""

    text: str = PROMPT_DESCRIPTION
    """
    The template for the prompt to generate descriptions for intents.
    Should include placeholders for {intent_name} and {user_utterances}.
    - `{intent_name}` will be replaced with the name of the intent.
    - `{user_utterances}` will be replaced with the user utterances related to the intent.
    - (optionally) `{regex_patterns}` will be replaced with the regular expressions that match user utterances.
    """

    @classmethod
    @field_validator("text")
    def check_valid_prompt(cls, value: str) -> str:
        """Validate the prompt description template.

        Args:
            value: The prompt description template.

        Returns:
            The validated prompt description template.
        """
        if value.find("{intent_name}") == -1 or value.find("{user_utterances}") == -1:
            text_error = (
                "The 'prompt_description' template must properly "
                "include {intent_name} and {user_utterances} placeholders."
            )
            raise ValueError(text_error)
        return value
