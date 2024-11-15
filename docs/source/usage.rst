Usage
=====

Установка
---------

1. Скопировать проект:

.. code-block:: bash

    git clone https://github.com/voorhs/AutoIntent.git
    cd AutoIntent

2. Установить пакет:

.. code-block:: bash

    pip install .


Оптимизация
-----------

В текущей alpha-версии оптимизацию можно запустить командой ``autointent``:

Примеры использования:

.. code-block:: bash

    autointent data.train_path=default-multiclass
    autointent data.train_path=default-multilabel hydra.job_logging.root.level=INFO
    autointent data.train_path=data/intent_records/ac_robotic_new.json \
        data.force_multilabel=true \
        logs.dirpath=experiments/multiclass_as_multilabel/ \
        logs.run_name=robotics_new_testing \
        augmentation.regex_sampling=10 \
        augmentation.multilabel_generation_config="[0, 4000, 1000]"  # currently doesn't work, omit this line
        # currently doesn't work due to problems with to_multilabel when dataset contains only regexp but no utterances
    autointent data.train_path=data/intent_records/ac_robotic_new.json \
            data.test_path=data/intent_records/ac_robotic_val.json \
            data.force_multilabel=true \
            augmentation.regex_sampling=20
    autointent data.train_path=default-multiclass \
            data.test_path=data/intent_records/banking77_test.json \
            seed=42
