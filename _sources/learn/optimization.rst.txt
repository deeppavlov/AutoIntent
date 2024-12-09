Optimization
============

In this section, you will learn how hyperparameter optimization works in our library.

Pipeline
--------

The entire process of configuring a classifier in our library is divided into sequential steps (:ref:`and that's why <key-stages>`):

1. Selecting an embedder (EmbeddingNode)
2. Selecting a classifier (ScoringNode)
3. Selecting a decision rule (PredictionNode)

Each step has its own set of hyperparameters. To theoretically guarantee finding the ideal configuration through exhaustive search, it is necessary to check every element of the Cartesian product of the hyperparameter sets of these steps (grid search). In practice, achieving this is usually impossible because the number of combinations is too large.

Greedy Strategy
---------------

This is one of the ways to solve the problem of an overwhelming number of combinations. In our case, the greedy optimization algorithm is as follows:

1. Iterate through the hyperparameters of the embedder and fix the best one.
2. Iterate through the hyperparameters of the classifier and fix the best one.
3. Iterate through the hyperparameters of the decision rule and fix the best one.

This algorithm checks fewer combinations, which speeds up the process. To implement such an algorithm, it is necessary to be able to evaluate the quality of not only the final prediction of the entire pipeline but also its intermediate predictions. The main drawback of this approach is that the decisions made are optimal only locally, not globally. The metrics for evaluating intermediate predictions are only a proxy signal for the quality of the final prediction.

This approach has been available in our library since release v0.0.1.

Random Search
-------------

A simpler strategy is to take a random subset of the full search space (random grid search). A straightforward strategy is to iterate through all combinations in random order until a certain time budget is exhausted.

This approach is less intelligent than the greedy strategy because, at any moment during the random combination search, poor embedders or any other bad parameters might keep appearing, despite they have been tested already. The greedy strategy would have eliminated such embedders at the beginning and not revisited them. On the other hand, random search, by its nature, does not rely on any local decisions.

The implementation of this optimization method is planned for release v0.1.0.

Bayesian Optimization
---------------------

This is similar to random search over a subset, but during the search, we attempt to model the probabilistic space of hyperparameters. This allows us to avoid repeating hyperparameter values that have previously performed poorly. The search itself aims to balance exploration and exploitation.

This approach is more sophisticated and can lead to better results by intelligently exploring the hyperparameter space.

The implementation of Bayesian optimization is planned for release v0.1.0.
