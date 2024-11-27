Quick Start
===========

Installation
------------

AutoIntent можно установить с помощью менеджера пакетов ``pip``:

.. code-block:: bash

    git clone https://github.com/voorhs/AutoIntent.git
    cd AutoIntent
    pip install .

Библиотека совместима с Python 3.10+.

Data Format
-----------

Для работы с AutoIntent нужно привести обучающие данные в специальный формат. You need to provide a training split containing samples with utterances and labels, as shown below:

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

To use it with our Python API, you can yse our :class:`autointent.Dataset` object. 

.. code-block:: python

    from autointent import Dataset

    dataset = Dataset.from_dict({"train": [...]})

To load dataset from file system into python, :meth:`autointent.Dataset.from_json` method exists:

.. code-block:: python

    dataset = Dataset.from_json("/path/to/json")

AutoML goes brrr...
-------------------

Когда данные готовы, можно запускать построение оптимального классификатора из командной строки:

.. code-block:: bash

    autointent data.train_path="path/to/your/data.json"

Эта команда запустит поиск гиперпараметров в дефолтном пространстве поиска.

По итогу в текущей рабочей директории создастся папка ``runs``, в которой будет сохранен подобранный классификатор, готовый к инференсу. Больше про папку рана и что в нем сохраняется смотрите в гайде :doc:`guides/optimization_results`.

Аналогичные действия но в ограниченном режиме можно запустить с помощью Python API:

.. code-block:: python

    from autointent import PipelineOptimizer

    pipeline_optimizer = PipelineOptimizer.default(multilabel=False)
    pipeline_optimizer.fit(dataset)

Inference
---------

Чтобы применить построенный классификатор к новым данным, достаточно воспользоваться нашим Python API:

.. code-block:: python

    from autointent import InferencePipeline

    inference_pipeline = InferencePipeline.load("path/to/run/directory")
    utterances = ["123", "hello world"]
    prediction = inference_pipeline.predict(utterances)

Modular Approach
----------------

Если нет необходимости в переборе пайплайнов и гиперпараметров, можно импортировать методы классификации напрямую.

.. code-block:: python

    from autointent.modules import KNNScorer

    scorer = KNNScorer(embedder_name="sergeyzh/rubert-tiny-turbo", k=1)
    train_utterances = [
        "why is there a hold on my american saving bank account",
        "i am nost sure why my account is blocked",
        "why is there a hold on my capital one checking account",
    ]
    train_labels = [0, 2, 1]
    scorer.fit(train_utterances, train_labels)
    test_utterances = [
        "i think my account is blocked but i do not know the reason",
        "can you tell me why is my bank account frozen",
    ]
    scorer.predict(test_utterances)

Futher Reading
--------------

- Больше про работу с данными в AutoIntent читайте в нашем туториале :doc:`tutorials/index_data.`
- Про то, как устроена автоконфигурация в нашей библиотеке, смотрите в разделе :doc:`learn/optimization`.
- Больше про пространство поиска и как выбирать его кастомным смотрите в гайде :doc:`guides/search_space_configuration`. 
- Построить классификатор из данных можно и с помощью Python API. Больше об этом смотрите в наших туториалах про оптимизацию :doc:`tutorials/index_pipeline_optimization`. 
- Больше про возможные варианты инференса смотрите в нашем туториале :doc:`tutorials/index_pipeline_inference`
- Больше про использование методов классификации напрямую, смотрите в наших туториалах :doc:`tutorials/index_scoring_modules`, :doc:`tutorials/index_prediction_modules`.