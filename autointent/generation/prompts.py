"""Prompt description."""

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
regexp patterns:
(activate.*card)|(start.*using.*card)
description:
User wants to activate his card.

name:
beneficiary_not_allowed
user utterances:

regexp patterns:
(not.*allowed.*beneficiary)|(cannot.*add.*beneficiary)
description:
User wants to know why his beneficiary is not allowed.

name:
vacation_registration
user utterances:
как оформить отпуск
в какие даты надо оформить отпуск
как запланировать отпуск
regexp patterns:

description:
Пользователь спрашивает про оформление отпуска.

name:
{intent_name}
user utterances:
{user_utterances}
regexp patterns:
{regexp_patterns}
description:

"""
