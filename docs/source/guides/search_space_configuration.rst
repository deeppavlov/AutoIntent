Search Space Configuration
==========================

In this guide, you will learn how to configure a custom hyperparameter search space.

Python API
##########

.. note::

    Before reading this guide, we recommend familiarizing yourself with the sections :doc:`../concepts` and :doc:`../learn/optimization`.

Optimization Module
-------------------

To set up the optimization module, you need to create the following dictionary:

.. code-block:: python

    knn_module = {
        "module_type": "knn",
        "k": [1, 5, 10, 50],
        "embedder_name": [
            "avsolatorio/GIST-small-Embedding-v0",
            "infgrad/stella-base-en-v2"
        ]
    }

The ``module_type`` field specifies the name of the module. You can find the names, for example, in :py:data:`autointent.modules.SCORING_MODULES_MULTICLASS`.

All fields except ``module_type`` are lists that define the search space for each hyperparameter. If you omit them, the default set of hyperparameters will be used during auto-configuration:

.. code-block:: python

    linear_module = {"module_type": "linear"}

Optimization Node
-----------------

To set up the optimization node, you need to create a list of modules and specify the metric for optimization:

.. code-block:: python

    scoring_node = {
        "node_type": "scoring",
        "metric_name": "scoring_roc_auc",
        "search_space": [
            knn_module,
            linear_module,
        ]
    }

Search Space
------------

The search space for the entire pipeline looks approximately like this:

.. code-block:: python

    search_space = [
        {
            "node_type": "retrieval",
            "metric": "retrieval_hit_rate",
            "search_space": [
                {
                    "module_type": "vector_db",
                    "k": [10],
                    "embedder_name": [
                        "avsolatorio/GIST-small-Embedding-v0",
                        "infgrad/stella-base-en-v2"
                    ]
                }
            ]
        },
        {
            "node_type": "scoring",
            "metric": "scoring_roc_auc",
            "search_space": [
                {
                    "module_type": "knn",
                    "k": [1, 3, 5, 10],
                    "weights": ["uniform", "distance", "closest"]
                },
                {
                    "module_type": "linear"
                },
                {
                    "module_type": "dnnc",
                    "cross_encoder_name": [
                        "BAAI/bge-reranker-base",
                        "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    ],
                    "k": [1, 3, 5, 10]
                }
            ]
        },
        {
            "node_type": "prediction",
            "metric": "prediction_accuracy",
            "search_space": [
                {
                    "module_type": "threshold",
                    "thresh": [0.5]
                },
                {
                    "module_type": "argmax"
                }
            ]
        }
    ]

Start Auto Configuration
------------------------

.. code-block:: python

    from autointent.pipeline import PipelineOptimizer

    pipeline_optimizer = PipelineOptimizer.from_dict(search_space)
    pipeline_optimizer.fit(dataset)

CLI
###

Yaml Format
-----------

YAML (YAML Ain't Markup Language) is a human-readable data serialization standard that is often used for configuration files and data exchange between languages with different data structures. It serves similar purposes as JSON but is much easier to read.

Here's an example YAML file:

.. code-block:: yaml

    database:
      host: localhost
      port: 5432
      username: admin
      # this is a comment
      password: secret

    counts:
    - 10
    - 20
    - 30

    literal_counts: [10, 20, 30]

    users:
    - name: Alice
      age: 30
      email: alice@example.com
    - name: Bob
      age: 25
      email: bob@example.com

    settings:
    debug: true
    timeout: 30

Explanation:

- the whole file represents a dictionary with keys ``database``, ``counts``, ``users``, ``settings``, ``debug``, ``timeout``
- ``database`` itself is a dictionary with keys ``host``, ``port``, and so on
- ``counts`` is a list (Python ``[10, 20, 30]``)
- ``literal_counts`` is a list too
- ``users`` is a list of dictionaries

Start Auto Configuration
------------------------

To set up the search space for optimization from the command line, you need to...
