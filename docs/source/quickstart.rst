Quickstart
==========

Welcome to AutoIntent! This guide will get you up and running with intent classification in just a few minutes.

What is AutoIntent?
-------------------

AutoIntent is a powerful AutoML library for intent classification that automatically finds the best model architecture and hyperparameters for your text classification tasks. Whether you're building chatbots or text analysis pipelines, AutoIntent simplifies the process of creating high-performance intent classifiers.

Key Features
------------

* ✨ **AutoML Pipeline**: Automated model selection and hyperparameter optimization
* 🔧 **Modular Design**: Use individual components or the full pipeline
* 📊 **Multiple Algorithms**: Support for classical neural networks, transformers, and traditional ML methods
* 📈 **Experiment Tracking**: Built-in support for Weights & Biases, TensorBoard and CodeCarbon

Installation
------------

Basic Installation
..................

AutoIntent is compatible with Python 3.10+. For core functionality:

.. code-block:: bash

    pip install autointent

With Experiment Tracking
........................

To include experiment tracking capabilities:

.. code-block:: bash

    pip install autointent[wandb,codecarbon]

Development Installation
........................

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/voorhs/AutoIntent.git
    cd AutoIntent
    pip install .

Quick Example
-------------

Here's a complete example that demonstrates AutoIntent's capabilities:

.. testcode:: python

    from autointent import Dataset, Pipeline

    # Prepare your data
    data = {
        "train": [
            {"utterance": "I want to check my account balance", "label": 0},
            {"utterance": "How do I transfer money?", "label": 1},
            {"utterance": "What's my current balance?", "label": 0},
            {"utterance": "I need to send money to my friend", "label": 1},
            {"utterance": "Can you help me make a payment?", "label": 1},
            {"utterance": "Show me my transaction history", "label": 0},
            {"utterance": "Can you show me my account details?", "label": 0},
            {"utterance": "I want to send funds to someone", "label": 1},
            {"utterance": "What is my available balance?", "label": 0},
            {"utterance": "How can I make a transfer?", "label": 1},
            {"utterance": "Please help me with a payment", "label": 1},
            {"utterance": "I need to view my recent transactions", "label": 0}
        ],
        "validation": [
            {"utterance": "Display my account info", "label": 0},
            {"utterance": "I want to transfer funds", "label": 1}
        ]
    }

    # Load data into AutoIntent
    dataset = Dataset.from_dict(data)

    # Initialize and train the AutoML pipeline
    pipeline = Pipeline.from_preset("classic-light")
    pipeline.fit(dataset)

    # Make predictions on new data
    predictions = pipeline.predict([
        "What is my available balance?",
        "Transfer money to John"
    ])

That's it! AutoIntent will automatically find the best model for your data.

Data Format
-----------

AutoIntent expects your data in a simple dictionary format with train/validation/test splits:

Single-Label Classification
...........................

.. code-block:: python

    data = {
        "train": [
            {"utterance": "Hello, how are you?", "label": 0},
            {"utterance": "Book a flight to Paris", "label": 1},
            {"utterance": "What's the weather like?", "label": 2}
        ],
        "validation": [  # Optional
            {"utterance": "Hi there!", "label": 0}
        ],
        "test": [  # Optional but highly recommended
            {"utterance": "Good morning", "label": 0}
        ]
    }

Multi-Label Classification
..........................

For multi-label tasks, use lists of 0s and 1s:

.. code-block:: python

    data = {
        "train": [
            {"utterance": "Book urgent flight to Paris", "label": [1, 0, 1]},  # booking=1, weather=0, urgent=1
            {"utterance": "What's the weather?", "label": [0, 1, 0]}
        ]
    }

Loading Data
............

.. code-block:: python

    from autointent import Dataset

    # From dictionary
    dataset = Dataset.from_dict(data)
    
    # From JSON file
    dataset = Dataset.from_json("/path/to/your/data.json")
    
    # From Hugging Face Hub
    dataset = Dataset.from_hub("your-username/your-dataset")

AutoML Training
---------------

AutoIntent provides several preset configurations optimized for different scenarios:

.. code-block:: python

    from autointent import Pipeline

    # Our quick and accurate SoTA
    pipeline = Pipeline.from_preset("classic-light")

    # If you have more training time
    pipeline = Pipeline.from_preset("classic-heavy")

    # Experimental preset with fine-tuning methods
    pipeline = Pipeline.from_preset("transformers-light")

    # Train the pipeline
    pipeline.fit(dataset)

Available Presets
.................

- ``classic-light``: Fast training with traditional ML methods
- ``classic-heavy``: Comprehensive search with traditional methods
- ``nn-medium``: Classic neural network-based approaches (RNN, CNN)
- ``nn-heavy``: Comprehensive neural network optimization
- ``transformers-light``: Transformer models with limited search
- ``transformers-no-hpo``: Transformer models without hyperparameter optimization
- ``zero-shot-openai``: Zero-shot classification using OpenAI models
- ``zero-shot-transformers``: Zero-shot classification using transformer models

Making Predictions
-------------------

Once trained, use your pipeline for inference:

.. code-block:: python

    # Batch predictions
    results = pipeline.predict([
        "What's my account balance?",
        "Transfer $100 to John",
        "Show me recent transactions"
    ])


Direct Module Usage
-------------------

For more control, use individual components without AutoML:

.. testcode:: python

    from autointent.modules import KNNScorer

    # Initialize a specific scorer
    scorer = KNNScorer(
        embedder_config="sentence-transformers/all-MiniLM-L6-v2",
        k=3
    )

    # Train on your data
    train_utterances = [
        "Check my account balance",
        "Transfer money to account",
        "Show transaction history"
    ]
    train_labels = [0, 1, 0]
    
    scorer.fit(train_utterances, train_labels)

    # Make predictions
    predictions = scorer.predict([
        "What's my current balance?",
        "Send money to my friend"
    ])

Available Modules
.................

- **Scoring**: :class:`autointent.modules.KNNScorer`, :class:`autointent.modules.BertScorer`, :class:`autointent.modules.SklearnScorer`, :class:`autointent.modules.CatBoostScorer`
- **Decision**: :class:`autointent.modules.ArgmaxDecision`,  :class:`autointent.modules.TunableDecision`, :class:`autointent.modules.AdaptiveDecision`

See more at API reference  :doc:`Modules <autoapi/autointent/modules/index>`.

Next Steps
----------

🚀 **Ready to dive deeper?**

- **Concepts**: Learn about :doc:`concepts` and AutoIntent's architecture
- **Tutorials**: Follow our step-by-step guides in :doc:`user_guides`
- **Advanced Usage**: Explore custom configurations and advanced features
- **Examples**: Check out real-world examples in our `GitHub repository <https://github.com/voorhs/AutoIntent>`_

🛠️ **Need Help?**

- Report issues on our `GitHub Issues <https://github.com/voorhs/AutoIntent/issues>`_
- Join our community discussions
- Check out the full API reference

Happy intent classification! 🎯