Dialogue Systems
================

In this section, you will get acquainted with the basics of building dialogue systems.

Intents
-------

A dialogue system, in a broad sense, is a textual interface for interacting with a service (be it a food ordering service or a service for obtaining information about a bank account). Typically, the service supports a finite number of API methods that are invoked during the dialogue with the user. To determine which method is needed at a given moment in the dialogue, intent classifiers are used. If we reason in terms of machine learning, this is a text classification task.

A good intent classifier should consider the specifics of the dialogue system creation task:

- Domain multiplicity. The number of API methods can be large enough to train the classifier yourself.
- Detection of out-of-domain examples. It is necessary to handle cases where the user expresses unsupported intents.
- Intent multiplicity. At one point in the dialogue, for some tasks, several complementary intents may arise at once, and then, in terms of machine learning, the task reduces to multilabel classification.
- A vast set of existing methods and their hyperparameters. As they say, "for this task, you can go through many hyperparameters, and you will be going through these hyperparameters."
- Scarcity of the training sample. Collecting a diverse sample of examples of replicas and even more so of entire dialogues is quite difficult.
- Using ML classifiers together with a rule-based approach.

Four out of five problems listed in this list are solved by using the AutoIntent library!

Slots
-----

.. todo::

    someday


Script
------

.. todo::

    someday
    