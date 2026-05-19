.. _adversarial_human_like_augmentation:

Adversarial human-like augmentation
====================================

This tutorial covers :py:class:`autointent.generation.utterances.HumanUtteranceGenerator` together with :py:class:`autointent.generation.utterances.CriticHumanLike`. The generator proposes paraphrases of training utterances; the critic asks an LLM to label each candidate as ``human`` or ``generated``. Candidates classified as ``generated`` are rejected and refined in a loop until the critic accepts them (or retries are exhausted).

.. warning::

   This path is **experimental** and may hurt data quality if the critic or base model mis-judges natural text. Use small ``n_final_per_class`` values first and inspect outputs.

How it fits together
--------------------

- **Generator** — :py:class:`autointent.generation.Generator` wraps your chat/structured-output API (OpenAI-compatible).
- **CriticHumanLike** — builds a JSON-schema prompt so the LLM returns ``reasoning`` and ``label`` (``human`` \| ``generated``); :py:meth:`~autointent.generation.utterances.CriticHumanLike.is_human` returns whether the utterance passed.
- **HumanUtteranceGenerator** — orchestrates rewrite attempts per intent; :py:meth:`~autointent.generation.utterances.HumanUtteranceGenerator.augment` can append accepted samples back into a chosen split (default: train).

Installation
------------

Install the OpenAI-backed generator extra (the ``Generator`` wrapper loads the OpenAI client):

.. code-block:: bash

    pip install "autointent[openai]"

Set ``OPENAI_API_KEY`` (and optional base URL) as required by your deployment. No separate DSPy extra is needed for this augmentation path.

Minimal sketch
--------------

.. code-block:: python

    from autointent import Dataset
    from autointent.generation import Generator
    from autointent.generation.utterances import CriticHumanLike, HumanUtteranceGenerator

    dataset = Dataset.from_dict({...})  # your train split, with intent names if you use them in prompts

    llm = Generator(model_name="gpt-4o-mini")
    critic = CriticHumanLike(generator=llm)
    augmenter = HumanUtteranceGenerator(generator=llm, critic=critic, async_mode=False)

    new_samples = augmenter.augment(dataset, split_name="train", n_final_per_class=3)

See the API reference for full argument lists (:py:class:`~autointent.generation.utterances.HumanUtteranceGenerator`, :py:class:`~autointent.generation.utterances.CriticHumanLike`).
