from pydantic import BaseModel, field_validator

from autointent.generation.prompts import PROMPT_DESCRIPTION


class PromptDescription(BaseModel):
    text: str = PROMPT_DESCRIPTION

    @classmethod
    @field_validator("text")
    def check_valid_prompt(cls, value: str) -> str:
        if value.find("{intent_name}") == -1 or value.find("{user_utterances}") == -1:
            text_error = (
                "The 'prompt_description' template must properly "
                "include {intent_name} and {user_utterances} placeholders."
            )
            raise ValueError(text_error)
        return value
