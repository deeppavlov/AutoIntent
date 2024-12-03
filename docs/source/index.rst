AutoIntent documentation
========================

**AutoIntent** is an open source tool for automatic configuration of a text classification pipeline for intent prediction.

.. note::

   This project is under active development.

The task of intent detection is one of the main subtasks in creating task-oriented dialogue systems, along with scriptwriting and slot filling. AutoIntent project offers users the following:

- A convenient library of methods for intent classification that can be used in a sklearn-like "fit-predict" format.
- An AutoML approach to creating classifiers, where the only thing needed is to upload a set of labeled data.

Example of building an intent classifier in a couple of lines of code:

.. code-block:: python

   from autointent import PipelineOptimizer, InferencePipeline, Dataset

   dataset = Dataset.from_json("/path/to/json")
   pipeline_optimizer = PipelineOptimizer.default(multilabel=False)
   pipeline_optimizer.fit(dataset)
   pipeline_optimizer.dump()

   inference_pipeline = InferencePipeline.load("/path/to/run")
   inference_pipeline.predict(["Hello, World!"])

We recommend you to begin your exploration of our library from the :doc:`quickstart` page.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   quickstart
   concepts
   tutorials
   guides/index
   learn/index
