Key Concepts
============

.. _key-search-space:

Optimization Search Space
-------------------------

The automatic selection of a classifier occurs through the iteration of hyperparameters within a certain *search space*. Conceptually, this search space is a dictionary where the keys are the names of the hyperparameters, and the values are lists. The hyperparameters act as the coordinate "axes" of the search space, and the values in the lists act as points on this axis.

.. _key-stages:

Classification Stages
---------------------

Intent classification can be divided into two stages: scoring and decision. Scoring involves predicting the probabilities of the presence of each intent in a given utterance. Prediction involves forming the final decision based on the provided probabilities.

.. _key-oos:

Out-of-domain utterances
------------------------

If we want to detect out-of-domain examples, it is necessary to set a probability threshold during the decision stage, at which the presence of some known intent can be asserted.

.. _key-nodes-modules:

Nodes and Modules
-----------------

The scoring or decision model, along with its hyperparameters that need to be iterated, is called an *optimization module*. A set of modules related to one optimization stage (scoring or decision) is called an *optimization node*.
