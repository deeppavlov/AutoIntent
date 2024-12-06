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

   from autointent import Pipeline, Dataset

   dataset = Dataset.from_json("/path/to/json")
   optimizer = Pipeline.default_optimizer(multilabel=False)
   pipeline.fit(dataset)
   pipeline.predict(["Hello, World!"])

Documentation Contents
----------------------

:doc:`Quickstart <quickstart>`
..............................

It is recommended to begin with the :doc:`quickstart` page. It contains overview of our capabilities.

:doc:`Key Concepts <concepts>`
..............................

This page contains basic information about the terms and concepts we use throughout our documentation.

:doc:`Tutorials <tutorials>`
............................

Newbie-friendly information on how to perform different tasks using our library.

:doc:`User Guides<guides/index>`
................................

Some advanced information

:doc:`Learn AutoIntent<learn/index>`
....................................

Some theoretical background

:doc:`API Reference <autoapi/autointent/index>`
...............................................

lotta stuff

.. toctree::
   :hidden:
   :maxdepth: 1

   quickstart
   concepts
   tutorials
   guides/index
   learn/index
   autoapi/autointent/index