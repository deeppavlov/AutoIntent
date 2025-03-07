.. _evolutionary_strategy_augmentation:

DSPY Augmentation
#################

This tutorial covers the implementation and usage of an evolutionary strategy to augment utterances using DSPy. It explains how DSPy is used, how the module functions, and how the scoring metric works.

.. contents:: Table of Contents
    :depth: 2

What is DSPy?
-------------

DSPy is a framework for optimizing and evaluating language models. It provides tools for defining signatures, optimizing modules, and measuring evaluation metrics. This module leverages DSPy to generate augmented utterances using an evolutionary approach.

How This Module Works
---------------------

This module applies an incremental evolutionary strategy for augmenting utterances. It generates new utterances based on a given dataset and refines them using an iterative process. The generated utterances are evaluated using a scoring mechanism that includes:

- **SemanticF1**: Measures how well the generated utterance matches the ground truth.
- **ROUGE-1 penalty**: Discourages excessive repetition.
- **Pipeline Decision Metric**: Assesses whether the augmented utterances improve model performance.

The augmentation process runs for a specified number of evolutions, saving intermediate models and optimizing the results.

Installation
------------

Ensure you have the required dependencies installed:

.. code-block:: bash

    pip install "autointent[dspy]"

Scoring Metric
--------------

The scoring metric consists of:

1. **SemanticF1 Score**:
   - Computes precision and recall between system-generated utterances and ground truth by LLM.
   - Uses DSPy’s `SemanticRecallPrecision` module.

2. **Repetition Factor (ROUGE-1 Penalty)**:
   - Measures overlap of words between the generated and ground truth utterances.
   - Ensures diversity in augmentation.

3. **Final Score Calculation**:
   - `Final Score = SemanticF1 * Repetition Factor`
   - A higher score means better augmentation.

Usage Example
-------------

Before running the following code, refer to the `LiteLLM documentation <https://docs.litellm.ai/docs/providers>`_ for proper model configuration.

.. code-block:: python

    import os
    os.environ["OPENAI_API_KEY"] = "your-api-key"

    from autointent import Dataset
    from autointent.custom_types import Split

    dataset = Dataset.from_hub("AutoIntent/clinc150_subset")
    evolver = DSPYIncrementalUtteranceEvolver(
        "openai/gpt-4o-mini"
    )

    augmented_dataset = evolver.augment(
        dataset,
        split_name=Split.TEST,
        n_evolutions=1,
        mipro_init_params={
            "auto": "light",
        },
        mipro_compile_params={
            "minibatch": False,
        },
    )

    augmented_dataset.to_csv("clinc150_dspy_augment.csv")
