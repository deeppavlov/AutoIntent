Quickstart
===========

Installation
------------

AutoIntent can be installed using the package manager ``pip``:

.. code-block:: bash

    git clone https://github.com/voorhs/AutoIntent.git
    cd AutoIntent
    pip install .

The library is compatible with Python 3.10+.

Data Format
-----------

To work with AutoIntent, you need to format your training data in a specific way. You need to provide a training split containing samples with utterances and labels, as shown below:

.. code-block:: json

    {
        "train": [
            {
                "utterance": "Hello!",
                "label": 0
            }
        ]
    }

For a multilabel dataset, the ``label`` field should be a list of integers representing the corresponding class labels.

To use it with our Python API, you can use our :class:`autointent.Dataset` object.

.. code-block:: python

    from autointent import Dataset

    dataset = Dataset.from_dict({"train": [...]})

To load a dataset from the file system into Python, the :meth:`autointent.Dataset.from_json` method exists:

.. code-block:: python

    dataset = Dataset.from_json("/path/to/json")

AutoML goes brrr...
-------------------

Once the data is ready, you can start building the optimal classifier from the command line:

.. code-block:: bash

    autointent data.train_path="path/to/your/data.json"

This command will start the hyperparameter search in the default :ref:`search space <key-search-space>`.

As a result, a ``runs`` folder will be created in the current working directory, which will save the selected classifier ready for inference. More about the run folder and what is saved in it can be found in the guide :doc:`guides/optimization_results`.

Similar actions but in a limited mode can be started using the Python API:

.. code-block:: python

    from autointent import PipelineOptimizer

    pipeline_optimizer = PipelineOptimizer.default(multilabel=False)
    pipeline_optimizer.fit(dataset)

Inference
---------

To apply the built classifier to new data, you can use our Python API:

.. code-block:: python

    from autointent import Pipeline

    pipeline = Pipeline.load("path/to/run/directory")
    utterances = ["123", "hello world"]
    prediction = pipeline.predict(utterances)

Modular Approach
----------------

If there is no need to iterate over pipelines and hyperparameters, you can import classification methods directly.

.. code-block:: python

    from autointent.modules import KNNScorer

    scorer = KNNScorer(embedder_name="sergeyzh/rubert-tiny-turbo", k=1)
    train_utterances = [
        "why is there a hold on my american saving bank account",
        "i am not sure why my account is blocked",
        "why is there a hold on my capital one checking account",
    ]
    train_labels = [0, 2, 1]
    scorer.fit(train_utterances, train_labels)
    test_utterances = [
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]
    scorer.predict(test_utterances)

Further Reading
---------------

- Get familiar with :doc:`concepts`.
- Learn more about working with data in AutoIntent in our tutorials :doc:`tutorials/index_data`.
- Learn about how auto-configuration works in our library in the section :doc:`learn/optimization`.
- Learn more about the search space and how to customize it in the tutorials :doc:`tutorials/index_pipeline_optimization`.
- You can also build a classifier from data using the Python API. Learn more about this in our optimization tutorials :doc:`tutorials/index_pipeline_optimization`.
- Learn more about using classification methods directly in our tutorials :doc:`tutorials/index_scoring_modules`, :doc:`tutorials/index_prediction_modules`.
