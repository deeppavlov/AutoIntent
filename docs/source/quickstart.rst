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

Эта команда запустит поиск гиперпараметров в дефолтном пространстве поиска. Больше про пространство поиска и как выбирать его кастомным смотрите в :doc:`search_space`. 

По итогу в текущей рабочей директории создастся папка ``runs``, в которой будет сохранен подобранный классификатор, готовый к инференсу. Больше про папку рана и что в нем сохраняется смотрите в :doc:`optimization_results`.

.. seealso::

    Построить классификатор из данных можно и с помощью Python API. Больше об этом смотрите в нашем туториале :doc:`tutorials/index_pipeline_optimization`. 

Inference
---------

Чтобы применить построенный классификатор к новым данным, достаточно воспользоваться нашим Python API:

.. code-block:: python

    from autointent import InferencePipeline

    inference_pipeline = InferencePipeline.load("path/to/run/directory")
    utterances = ["123", "hello world"]
    prediction = inference_pipeline.predict(utterances)

