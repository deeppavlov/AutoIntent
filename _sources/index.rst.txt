AutoIntent documentation
========================

**AutoIntent** is an open source tool for automatic configuration of a text classification pipeline for intent prediction.

.. note::

   This project is under active development.

The task of intent detection is one of the main subtasks in creating task-oriented dialogue systems, along with scriptwriting and slot filling. AutoIntent project offers users the following:

- A convenient library of methods for intent classification that can be used in a sklearn-like "fit-predict" format.
- An AutoML approach to creating classifiers, where the only thing needed is to upload a set of labeled data.

Example of building an intent classifier in a couple of lines of code:

.. testsetup::

   import importlib.resources as ires

   path_to_json = ires.files("tests.assets.data").joinpath("clinc_subset.json")

.. testcode::

   from autointent import Pipeline, Dataset

   dataset = Dataset.from_json(path_to_json)
   pipeline = Pipeline.default_optimizer(multilabel=False)
   pipeline.fit(dataset)
   pipeline.predict(["show me my latest recent transactions"])

.. testcleanup::

   import shutil
   from glob import glob
   for match in glob("vector_db*"):
      shutil.rmtree(match)

Documentation Contents
----------------------

:doc:`Quickstart <quickstart>`
..............................

It is recommended to begin with the :doc:`quickstart` page. It contains overview of our capabilities and basic instructions for working with our library.

:doc:`Key Concepts <concepts>`
..............................

Key terms and concepts we use throughout our documentation.

:doc:`User Guides<user_guides>`
................................

A series of notebooks that demonstrate in detail and comprehensively the capabilities of our library and how to use it.

:doc:`API Reference <autoapi/autointent/index>`
...............................................

Pay special attention to the sections :doc:`autoapi/autointent/modules/index` and :doc:`autoapi/autointent/metrics/index`.

:doc:`Learn AutoIntent<learn/index>`
....................................

Some theoretical background on dialogue systems and auto ML.


.. toctree::
   :hidden:
   :maxdepth: 1

   quickstart
   concepts
   user_guides
   learn/index
   autoapi/autointent/index