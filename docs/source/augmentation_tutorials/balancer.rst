.. _balancer_aug:

Balancing Datasets with DatasetBalancer
=======================================

This guide demonstrates how to use the :py:class:`autointent.generation.utterances.DatasetBalancer` class to balance class distribution in your datasets through LLM-based data augmentation. This method is a wrapper for more simple method :py:class:`autointent.generation.utterances.UtteranceGenerator`.

.. contents:: Table of Contents
    :depth: 2

Why Balance Datasets?
---------------------

Imbalanced datasets can lead to biased models that perform well on majority classes but poorly on minority classes. DatasetBalancer helps address this issue by generating additional examples for underrepresented classes using large language models.

Creating a Sample Imbalanced Dataset
-----------------------------------

Let's create a small imbalanced dataset to demonstrate the balancing process:

.. code-block:: python

    from autointent import Dataset
    from autointent.generation.utterances.balancer import DatasetBalancer
    from autointent.generation.utterances.generator import Generator
    from autointent.generation.chat_templates import EnglishSynthesizerTemplate

    # Create a simple imbalanced dataset
    sample_data = {
        "intents": [
            {"id": 0, "name": "restaurant_booking", "description": "Booking a table at a restaurant"},
            {"id": 1, "name": "weather_query", "description": "Checking weather conditions"},
            {"id": 2, "name": "navigation", "description": "Getting directions to a location"},
        ],
        "train": [
            # Restaurant booking examples (5)
            {"utterance": "Book a table for two tonight", "label": 0},
            {"utterance": "I need a reservation at Le Bistro", "label": 0},
            {"utterance": "Can you reserve a table for me?", "label": 0},
            {"utterance": "I want to book a restaurant for my anniversary", "label": 0},
            {"utterance": "Make a dinner reservation for 8pm", "label": 0},

            # Weather query examples (3)
            {"utterance": "What's the weather like today?", "label": 1},
            {"utterance": "Will it rain tomorrow?", "label": 1},
            {"utterance": "Weather forecast for New York", "label": 1},

            # Navigation example (1)
            {"utterance": "How do I get to the museum?", "label": 2},
        ]
    }

    # Create the dataset
    dataset = Dataset.from_dict(sample_data)

Setting up the Generator and Template
------------------------------------

DatasetBalancer requires two main components:

1. A :py:class:`autointent.generation.Generator`` - responsible for creating new utterances using an LLM
2. A :py:class:`autointent.generation.chat_templates.EnglishSynthesizerTemplate` - defines the prompt format sent to the LLM

Let's set up these components:

.. code-block:: python

    # Initialize a generator (uses OpenAI API by default)
    generator = Generator()

    # Create a template for generating utterances
    template = EnglishSynthesizerTemplate(dataset=dataset, split="train")

Creating the DatasetBalancer
----------------------------

Now we can create our DatasetBalancer instance:

.. code-block:: python

    balancer = DatasetBalancer(
        generator=generator,
        prompt_maker=template,
        async_mode=False,  # Set to True for faster generation with async processing
        max_samples_per_class=5,  # Each class will have exactly 5 samples after balancing
    )

Checking Initial Class Distribution
----------------------------------

Let's examine the class distribution before balancing:

.. code-block:: python

    # Check the initial distribution of classes in the training set
    initial_distribution = {}
    for sample in dataset["train"]:
        label = sample[Dataset.label_feature]
        initial_distribution[label] = initial_distribution.get(label, 0) + 1

    print("Initial class distribution:")
    for class_id, count in sorted(initial_distribution.items()):
        intent = next(i for i in dataset.intents if i.id == class_id)
        print(f"Class {class_id} ({intent.name}): {count} samples")

    print(f"\nMost represented class: {max(initial_distribution.values())} samples")
    print(f"Least represented class: {min(initial_distribution.values())} samples")

Balancing the Dataset
---------------------

Now we'll use the DatasetBalancer to augment our dataset:

.. code-block:: python

    # Create a copy of the dataset
    dataset_copy = Dataset.from_dict(dataset.to_dict())

    # Balance the training split
    balanced_dataset = balancer.balance(
        dataset=dataset_copy,
        split="train",
        batch_size=2,  # Process generations in batches of 2
    )

Checking the Results
-------------------

Let's examine the class distribution after balancing:

.. code-block:: python

    # Check the balanced distribution
    balanced_distribution = {}
    for sample in balanced_dataset["train"]:
        label = sample[Dataset.label_feature]
        balanced_distribution[label] = balanced_distribution.get(label, 0) + 1

    print("Balanced class distribution:")
    for class_id, count in sorted(balanced_distribution.items()):
        intent = next(i for i in dataset.intents if i.id == class_id)
        print(f"Class {class_id} ({intent.name}): {count} samples")

    print(f"\nMost represented class: {max(balanced_distribution.values())} samples")
    print(f"Least represented class: {min(balanced_distribution.values())} samples")

Examining Generated Examples
---------------------------

Let's look at some examples of original and generated utterances for the navigation class,
which was the most underrepresented:

.. code-block:: python

    # Navigation class (Class 2)
    navigation_class_id = 2
    intent = next(i for i in dataset.intents if i.id == navigation_class_id)

    print(f"Examples for class {navigation_class_id} ({intent.name}):")

    # Original examples
    original_examples = [
        s[Dataset.utterance_feature] for s in dataset["train"] if s[Dataset.label_feature] == navigation_class_id
    ]
    print("\nOriginal examples:")
    for i, example in enumerate(original_examples, 1):
        print(f"{i}. {example}")

    # Generated examples
    all_examples = [
        s[Dataset.utterance_feature] for s in balanced_dataset["train"] if s[Dataset.label_feature] == navigation_class_id
    ]
    generated_examples = [ex for ex in all_examples if ex not in original_examples]
    print("\nGenerated examples:")
    for i, example in enumerate(generated_examples, 1):
        print(f"{i}. {example}")

Configuring the Number of Samples per Class
------------------------------------------

You can configure how many samples each class should have:

.. code-block:: python

    # To bring all classes to exactly 10 samples
    original_dataset = Dataset.from_dict(sample_data)
    exact_template = EnglishSynthesizerTemplate(dataset=original_dataset, split="train")

    exact_balancer = DatasetBalancer(
        generator=generator,
        prompt_maker=exact_template,
        max_samples_per_class=10
    )

    # Balance to the level of the most represented class
    max_template = EnglishSynthesizerTemplate(dataset=original_dataset, split="train")

    max_balancer = DatasetBalancer(
        generator=generator,
        prompt_maker=max_template,
        max_samples_per_class=None  # Will use the count of the most represented class
    )

Tips for Effective Dataset Balancing
-----------------------------------

1. **Quality Control**: Always review a sample of generated utterances to ensure quality.
2. **Template Selection**: Different templates may work better for different domains.
3. **Model Selection**: Larger models generally produce higher quality utterances.
4. **Batch Size**: Increase batch size for faster generation if your hardware allows.
5. **Validation**: Test your model on both original and augmented data to ensure it generalizes well.
