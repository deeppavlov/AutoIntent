"""Prompt description configuration."""

from pydantic import BaseModel, field_validator

from autointent.generation.chat_templates import Message, Role

PROMPT_DESCRIPTION_SYSTEM = """
Your task is to write a description of the intent.

You are given the name of the intent, user intentions related to it. The description should be:
1) In declarative form.
2) No more than one sentence.
3) In the language in which the utterances.

Remember:
- Respond with just the description, no extra details.
- Keep in mind that either the names or user queries may not be provided.

For example:

Input:
name:
activate_my_card
user utterances:
- Please help me with my card. It won't activate.
- I tried but am unable to activate my card.
- I want to start using my card.

Output:
User wants to activate his card.

Input:
name:
beneficiary_not_allowed
user utterances:

Output:
User wants to know why his beneficiary is not allowed.
"""
PROMPT_DESCRIPTION_USER = """
name:
{intent_name}
user utterances:
{user_utterances}
"""


class PromptDescription(BaseModel):
    """Prompt description configuration."""

    system_text: str = PROMPT_DESCRIPTION_SYSTEM
    user_text: str = PROMPT_DESCRIPTION_USER
    """
    The template for the prompt to generate descriptions for intents.
    Should include placeholders for {intent_name} and {user_utterances}.
    - `{intent_name}` will be replaced with the name of the intent.
    - `{user_utterances}` will be replaced with the user utterances related to the intent.
    """

    @classmethod
    @field_validator("user_text")
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

    def to_messages(self, intent_name: str | None, utterances: list[str]) -> list[Message]:
        user_message_content = self.user_text.format(
            intent_name=intent_name,
            user_utterances="\n - ".join(utterances),
        )
        return [
            Message(role=Role.SYSTEM, content=self.system_text),
            Message(role=Role.USER, content=user_message_content),
        ]
