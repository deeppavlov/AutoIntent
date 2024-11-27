Search Space Configuration
==========================

В этом гайде вы узнаете как настраивать кастомное пространство поиска гипепараметров.

Python API
##########

.. note::

    Перед чтением данного гайда советуем ознамомиться с разделами :doc:`../concepts` и :doc:`../learn/greedy_optimization`.

Optimization Module
-------------------

Чтобы задать модуль оптимизации, достаточно создать следующий словарик:

.. code-block:: python

    knn_module = {
        "module_type": "knn",
        "k": [1, 5, 10, 50],
        "embedder_name": [
            "avsolatorio/GIST-small-Embedding-v0",
            "infgrad/stella-base-en-v2"
        ]
    }

В поле ``module_type`` указано название модуля. Названия можете посмотреть например в :py:data:`autointent.modules.SCORING_MODULES_MULTICLASS`.

Все поля, кроме ``module_type`` являются списками, задающими пространство поиска по каждому гиперпарметру. Если опустить их, то во время автоконфигурации будет использован дефолтный набор гиперпараметров:


.. code-block:: python

    linear_module = {"module_type": "linear"}

Optimization Node
-----------------

Чтобы задать узел оптимизации, нужно создать список модулей и задать метрику для оптимизации:

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

Серч спейс всего пайплайна выглядит примерно следующим образом:

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

YAML (YAML Ain't Markup Language) is a human-readable data serialization standard that is often used for configuration files and data exchange between languages with different data structures. It serves for similar purposes as JSON but much easier to read.

Here's example yaml file:

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

- the whole file represents dictionary with keys ``database``, ``counts``, ``users``, ``settings``, ``debug``, ``timeout``
- ``database`` itself is a dictionary with keys ``host``, ``port`` and so on
- ``counts`` is a list (python ``[10, 20, 30]``)
- ``literal_counts`` is a list too
- ``users`` is a list of dictionaries

Start Auto Configuration
------------------------

Чтобы задать серч спейс для оптимизации из командной строки, достаточно...