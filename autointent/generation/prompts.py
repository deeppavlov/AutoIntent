PROMPT_DESCRIPTION = """
Your task is to write a description of the intent.

You are given the name of the intent and user intentions related to it. The description should be:
1) in declarative form
2) no more than one sentence
3) in the language in which the utterances and the name are written.

Remember:
Respond with just the description, no extra details.
Keep in mind that either the names or user queries may not be provided.

For example:

name:
activate_my_card
user utterances:
Please help me with my card. It won't activate.
I tried but am unable to activate my card.
I want to start using my card.
description:
user wants to activate his card

name:
beneficiary_not_allowed
user utterances:

description:
user want to know why his beneficiary is not allowed

name:
оформление_отпуска
user utterances:
как оформить отпуск
в какие даты надо оформить отпуск
как запланировать отпуск
description:
пользователь спрашивает про оформление отпуска

name:
{intent_name}
user utterances:
{user_utterances}
description:
"""
