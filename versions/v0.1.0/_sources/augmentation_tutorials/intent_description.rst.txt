.. _intent_description_generation:

Intent Description Generation
#############################

This documentation covers the implementation and usage of the Intent Description Generation module. It explains the function of the module, the underlying mechanisms, and provides examples of usage.

The approach used in this module is based on the paper `Exploring Description-Augmented Dataless Intent Classification <https://arxiv.org/pdf/2407.17862>`_.

.. contents:: Table of Contents
    :depth: 2

Overview
--------

The Intent Description Generation module is designed to automatically generate detailed and coherent descriptions of intents using large language models (LLMs). It enhances datasets by creating human-readable explanations for intents, supplemented by examples (utterances) and regex patterns.

How the Module Works
--------------------

The module leverages prompt engineering to interact with LLMs, creating structured intent descriptions that are suitable for documentation, user interaction, and training purposes. Each generated description includes:

- **Intent Name**: Clearly identifies the intent.
- **Examples (User Utterances)**: Demonstrates real-world user inputs.
- **Regex Patterns**: Highlights relevant regex patterns associated with the intent.

The module uses a templated approach, defined through `PromptDescription`, to maintain consistency and clarity across descriptions.

Installation
------------

Ensure you have the necessary dependencies installed:

.. code-block:: bash

    pip install autointent openai

Usage
-----

Here's an example demonstrating how to generate intent descriptions:

.. code-block:: python

    import openai
    from autointent import Dataset
    from autointent.generation.intents import generate_descriptions
    from autointent.generation.chat_templates import PromptDescription

    client = openai.AsyncOpenAI(
        api_key="your-api-key"
    )

    dataset = Dataset.from_hub("AutoIntent/clinc150_subset")

    prompt = PromptDescription(
        text="Describe intent {intent_name} with examples: {user_utterances} and patterns: {regex_patterns}",
    )

    enhanced_dataset = generate_descriptions(
        dataset=dataset,
        client=client,
        prompt=prompt,
        model_name="gpt4o-mini",
    )

    enhanced_dataset.to_csv("enhanced_clinc150.csv")

Prompt Customization
--------------------

The `PromptDescription` can be customized to better fit specific requirements. It uses the following placeholders:

- ``{intent_name}``: The name of the intent being described.
- ``{user_utterances}``: Example utterances related to the intent.
- ``{regex_patterns}``: Associated regular expression patterns.

Adjusting the prompt allows tailoring descriptions to different contexts or detail levels.

Model Selection
---------------

This module supports various LLMs available through OpenAI-compatible APIs. Configure your preferred model via the `model_name` parameter. Refer to your LLM provider’s documentation for available models.

Recommended models include:

- ``gpt4o-mini`` (for balanced performance and efficiency)
- ``gpt-4`` (for maximum descriptive quality)

API Integration
---------------

Ensure your OpenAI-compatible client is properly configured with an API endpoint and key:

.. code-block:: python

    client = openai.AsyncOpenAI(
        base_url="your-api-base-url",
        api_key="your-api-key"
    )
