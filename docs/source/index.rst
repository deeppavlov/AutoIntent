AutoIntent documentation
========================

**AutoIntent** is an open source tool for automatic configuration of text classification pipelines, with specialized support for intent prediction.

.. note::

   This project is under active development.

The task of intent detection is one of the main subtasks in creating task-oriented dialogue systems, along with scriptwriting and slot filling. While AutoIntent is particularly well-suited for intent detection, it can be applied to any text classification problem, including sentiment analysis, topic classification, document categorization, and other NLP tasks.

AutoIntent project offers users the following:

- A convenient library of methods for intent classification that can be used in a sklearn-like "fit-predict" format.
- An AutoML approach to creating classifiers, where the only thing needed is to upload a set of labeled data.

Example of building an intent classifier in a couple of lines of code:

.. testsetup::

   import importlib.resources as ires

   path_to_json = ires.files("tests.assets.data").joinpath("clinc_subset.json")

.. testcode::

   from autointent import Pipeline, Dataset

   dataset = Dataset.from_json(path_to_json)
   pipeline = Pipeline.from_preset("classic-light")
   pipeline.fit(dataset)
   pipeline.predict(["show me my latest recent transactions"])

.. testcleanup::

   import shutil
   from glob import glob
   for match in glob("vector_db*"):
      shutil.rmtree(match)

Documentation Guide
-------------------

Getting Started
...............

:doc:`🚀 Quickstart <quickstart>`
   Jump right in! Install AutoIntent and build your first text classifier in minutes. Perfect for users who want to get up and running quickly with practical examples.

:doc:`📚 Key Concepts <concepts>`
   Essential terminology and concepts used throughout AutoIntent. Understanding these will help you navigate the documentation and make the most of the library's features.

In-Depth Learning
.................

:doc:`📖 User Guides <user_guides>`
   Comprehensive tutorials and examples that walk you through AutoIntent's capabilities step-by-step. These hands-on guides cover everything from basic usage to advanced techniques.

:doc:`🎓 Learn AutoIntent <learn/index>`
   Dive deeper into the theory behind AutoIntent. Learn about dialogue systems, AutoML principles, and the science that powers intelligent text classification.

Reference
.........

:doc:`🔧 API Reference <autoapi/autointent/index>`
   Complete technical documentation for all classes, methods, and functions. Essential reference for developers integrating AutoIntent into their applications.
   
   Key section: :doc:`Modules <autoapi/autointent/modules/index>`


.. toctree::
   :hidden:
   :maxdepth: 1

   quickstart
   concepts
   user_guides
   learn/index
   autoapi/autointent/index