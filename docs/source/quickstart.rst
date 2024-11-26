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

Для работы с AutoIntent нужно привести обучающие данные в специальный формат.

.. todo::

    Написать когда закончится рефакторинг датасетов

AutoML goes brrr...
-------------------

Когда данные готовы, можно запускать построение оптимального классификатора.

Из командной строки:

.. code-block:: bash

    autointent data.train_path="path/to/your/data.json"

Эта команда запустит поиск гиперпараметров в дефолтном пространстве поиска. Больше про пространство поиска и как выбирать его кастомным смотрите в гайде :doc:`guides/search_space_configuration`. 

По итогу в текущей рабочей директории создастся папка ``runs``, в которой будет сохранен подобранный классификатор, готовый к инференсу. Больше про папку рана и что в нем сохраняется смотрите в гайде :doc:`guides/optimization_results`.

.. seealso::

    Построить классификатор из данных можно и с помощью Python API. Больше об этом смотрите в наших туториалах про оптимизацию :doc:`tutorials/index_pipeline_optimization`. 

Inference
---------

Чтобы применить построенный классификатор к новым данным, достаточно воспользоваться нашим Python API:

.. code-block:: python

    from autointent import InferencePipeline

    inference_pipeline = InferencePipeline.load("path/to/run/directory")
    utterances = ["123", "hello world"]
    prediction = inference_pipeline.predict(utterances)

.. seealso::

    Больше про возможные варианты инференса смотрите в нашем туториале :doc:`tutorials/index_pipeline_inference`

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

.. seealso::

    Больше про использование методов классификации напрямую, смотрите в наших туториалах :doc:`tutorials/index_scoring_modules`, :doc:`tutorials/index_prediction_modules`.
