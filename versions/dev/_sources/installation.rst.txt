.. _installation:

Installation
============

AutoIntent supports **Python** ``>=3.10,<3.15`` (see ``requires-python`` in the package metadata).

Since **v0.3.0**, ``pip install autointent`` installs only the **core** runtime: PyTorch, scikit-learn, Optuna, vector search, datasets, and other always-on dependencies. Heavier or integration-specific libraries are shipped as **optional extras**.

Minimal install
---------------

For the smallest footprint (core AutoML and modules that do not require optional stacks):

.. code-block:: bash

   pip install autointent

If a preset or module needs an extra, install it with ``pip install "autointent[<extra>]"`` or combine several names in the brackets (comma-separated).

Optional extras matrix
----------------------

The table below lists each published extra, what it is typically used for in AutoIntent, and the matching ``pip`` command.

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Extra
     - What it enables
     - Install command
   * - ``sentence-transformers``
     - SentenceTransformer embedders and pipelines that rely on them (for example many ``classic-*`` and embedding-based scorers).
     - ``pip install "autointent[sentence-transformers]"``
   * - ``catboost``
     - ``CatBoostScorer`` and CatBoost-based tuning paths (see the modules API).
     - ``pip install "autointent[catboost]"``
   * - ``transformers``
     - Hugging Face ``transformers`` models used by transformer presets and modules.
     - ``pip install "autointent[transformers]"``
   * - ``peft``
     - Parameter-efficient fine-tuning (LoRA and similar) used together with transformer presets.
     - ``pip install "autointent[peft]"``
   * - ``openai``
     - OpenAI API clients for OpenAI-backed embedders and the ``zero-shot-llm`` style workflows.
     - ``pip install "autointent[openai]"``
   * - ``dspy``
     - DSPy-based augmentation and related generation utilities.
     - ``pip install "autointent[dspy]"``
   * - ``wandb``
     - Weights & Biases experiment logging integration.
     - ``pip install "autointent[wandb]"``
   * - ``codecarbon``
     - CodeCarbon energy / emissions tracking during runs.
     - ``pip install "autointent[codecarbon]"``
   * - ``fastapi``
     - HTTP serving stack (FastAPI, Uvicorn, settings) for the AutoIntent server mode.
     - ``pip install "autointent[fastapi]"``
   * - ``fastmcp``
     - FastMCP-based MCP server integration.
     - ``pip install "autointent[fastmcp]"``
   * - ``opensearch``
     - OpenSearch client for OpenSearch-backed vector / retrieval integrations.
     - ``pip install "autointent[opensearch]"``
   * - ``vllm``
     - vLLM as an optional high-throughput inference backend where supported.
     - ``pip install "autointent[vllm]"``

Recommended bundles
-------------------

Combine extras in one install by listing them inside the brackets (no spaces inside the list).

**Classic embedding + gradient boosting pipeline** (typical ``classic-light`` / ``classic-heavy`` style workflows with SentenceTransformers and CatBoost):

.. code-block:: bash

   pip install "autointent[sentence-transformers,catboost]"

**Transformer presets** (``transformers-*`` presets and fine-tuning): install both Hugging Face transformers and PEFT:

.. code-block:: bash

   pip install "autointent[transformers,peft]"

**Zero-shot LLM preset** (OpenAI API):

.. code-block:: bash

   pip install "autointent[openai]"

**HTTP API server**:

.. code-block:: bash

   pip install "autointent[fastapi]"

**MCP server** (FastMCP):

.. code-block:: bash

   pip install "autointent[fastmcp]"

**Experiment tracking** (pick what you use):

.. code-block:: bash

   pip install "autointent[wandb]"
   pip install "autointent[codecarbon]"

``codecarbon`` vs ``fastmcp``
-----------------------------

``codecarbon`` and ``fastmcp`` are declared as **conflicting extras**. A single environment should not select both at once; pick the extra that matches your goal (emissions tracking **or** FastMCP), or use separate environments if you genuinely need both stacks isolated.

Development install from Git
----------------------------

Clone the upstream repository and use **uv** with the project ``Makefile`` (see ``CONTRIBUTING.md`` in the repository root for details):

.. code-block:: bash

   git clone https://github.com/deeppavlov/AutoIntent.git
   cd AutoIntent
   make install

This installs the full contributor dependency set (tests, typing, docs, and so on), not only the minimal PyPI ``autointent`` wheel dependencies.
