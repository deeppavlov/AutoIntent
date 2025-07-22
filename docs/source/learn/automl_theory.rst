AutoML and Hyperparameter Optimization
======================================

This section provides a deep dive into the theoretical foundations of automated machine learning (AutoML) and hyperparameter optimization as implemented in AutoIntent.

The Hyperparameter Optimization Problem
---------------------------------------

**The Core Problem**

Hyperparameter optimization is about finding the best configuration of settings that maximizes model performance. Think of it as searching through all possible combinations of hyperparameters (like learning rates, model sizes, regularization strengths) to find the combination that gives the best results on validation data.

The performance metric is typically estimated through cross-validation to avoid overfitting - we want configurations that work well on unseen data, not just the training data.

**The Challenge of Combinatorial Explosion**

In AutoIntent's three-stage pipeline, the total search space grows multiplicatively across all stages. If we have:

- 10 different embedding models to choose from
- 20 different scoring configurations 
- 5 different decision strategies

Then we have 10 × 20 × 5 = 1,000 total combinations. In realistic scenarios, this can easily exceed 1,000,000 configurations, making it impossible to test every combination within reasonable time and computational budgets.

Hierarchical Optimization Strategy
----------------------------------

AutoIntent addresses combinatorial explosion through a **hierarchical greedy optimization** approach that optimizes modules sequentially.

**Sequential Module Optimization**

The optimization proceeds in three stages, where each stage builds on the results of the previous one:

1. **Embedding Optimization**: First, find the best embedding model configuration by testing different models and settings, evaluating them using retrieval or classification metrics.

2. **Scoring Optimization**: Using the best embedding model from step 1, now optimize the scoring module by testing different classifiers (KNN, linear, neural networks, etc.) with various hyperparameters.

3. **Decision Optimization**: Using the best embedding and scoring combination from steps 1-2, optimize the decision module by finding optimal thresholds and decision strategies for final predictions.

**Proxy Metrics**

Each stage uses specialized proxy metrics that correlate with final performance:

- **Embedding Stage**: Retrieval metrics (NDCG, hit rate) or lightweight classification accuracy
- **Scoring Stage**: Classification metrics (F1, ROC-AUC) on validation data  
- **Decision Stage**: Threshold-specific metrics for multi-label/OOS scenarios

**Trade-offs**

- ✅ **Computational Efficiency**: Instead of testing all possible combinations (which grows exponentially), we only test combinations within each stage separately, making optimization much faster and more manageable.
- ✅ **Parallelization**: Each stage can be parallelized independently, allowing multiple configurations to be tested simultaneously.
- ⚠️ **Local Optimality**: May miss globally optimal combinations due to greedy choices - the best embedding might work better with a different scorer than the one we pick, but we won't discover this combination.

Tree-Structured Parzen Estimators (TPE)
----------------------------------------

AutoIntent uses Optuna's TPE algorithm for sophisticated hyperparameter optimization within each module. This is a form of Bayesian optimization that learns from previous trials to make smarter choices about which hyperparameters to try next.

**How TPE Works**

TPE builds two separate models:

- **Good Configuration Model**: Learns the distribution of hyperparameters that led to good performance (typically the top 25% of trials)
- **Bad Configuration Model**: Learns the distribution of hyperparameters that led to poor performance (the remaining 75% of trials)

The algorithm then suggests new hyperparameters by finding configurations that are likely under the "good" model but unlikely under the "bad" model. This naturally balances exploration (trying untested areas) with exploitation (focusing on promising regions).

**Benefits of TPE**

- **Smart Sampling**: After initial random trials, TPE makes increasingly informed decisions about which hyperparameters to try
- **Handles Different Parameter Types**: Works well with categorical, continuous, and integer parameters
- **Robust to Noisy Evaluations**: Can handle situations where the same hyperparameters might give slightly different results due to randomness
- **No Prior Knowledge Required**: Works without needing to specify complex relationships between parameters

Search Space Design
-------------------

**Parameter Types**

AutoIntent supports various hyperparameter types with appropriate sampling strategies:

AutoIntent supports several types of hyperparameters, each requiring different optimization strategies:

**Categorical Parameters**: These are discrete choices from a fixed set of options, like choosing between different model types ("knn", "linear", "bert") or activation functions ("relu", "tanh", "sigmoid"). The optimizer samples uniformly from the available choices.

**Continuous Parameters**: These are real-valued parameters like learning rates, regularization strengths, or temperature values. The optimizer can sample from uniform distributions (for parameters like dropout rates between 0.0 and 1.0) or log-uniform distributions (for parameters like learning rates that work better on logarithmic scales).

**Integer Parameters**: These are whole number parameters like the number of neighbors in KNN, hidden dimensions in neural networks, or batch sizes. The optimizer can specify step sizes and bounds to ensure valid configurations.

**Conditional Parameters**: Some parameters only make sense when certain other parameters have specific values. For example, LoRA-specific parameters (like lora_alpha and lora_r) only apply when the model type is "lora". AutoIntent handles these dependencies automatically in the search space configuration.


**Search Space Configuration**

.. code-block:: yaml

   search_space:
     - node_type: scoring
       target_metric: scoring_f1
       search_space:
         - module_name: knn
           k:
             low: 1
             high: 20
           weights: [uniform, distance, closest]
         - module_name: linear
           cv: [3, 5, 10]

Cross-Validation and Data Splitting
-----------------------------------

**Validation Schemes**

AutoIntent supports multiple validation strategies to ensure robust hyperparameter selection:

**Hold-out Validation (HO)**

Split data into training and validation sets once. Train the model on the training set and evaluate performance on the validation set. This gives a single performance score for each hyperparameter configuration.

**Cross-Validation (CV)**

Split data into K folds (typically 3-5). For each fold, train on the remaining folds and validate on the current fold. Average the performance scores across all K folds to get a more robust estimate of how well the hyperparameters work.

**Stratified Splitting**

For imbalanced datasets, AutoIntent uses stratified sampling to maintain class distributions:

.. code-block:: python

   from autointent.configs import DataConfig
   
   data_config = DataConfig(
       scheme="cv",           # Cross-validation
       n_folds=5,             # 5-fold CV
       validation_size=0.2,   # 20% for validation in HO
       separation_ratio=0.5   # Prevent data leakage between modules
   )

**Data Leakage Prevention**

The ``separation_ratio`` parameter prevents information leakage between scoring and decision modules by using different data subsets for each stage.

**Hyperparameter Bounds**

Search spaces include reasonable bounds to prevent extreme configurations:

.. code-block:: yaml

   learning_rate:
     low: 1.0e-5    # Prevent too slow learning
     high: 1.0e-2   # Prevent instability
     log: true      # Log-uniform sampling

Multi-Objective Optimization Considerations
--------------------------------------------

While AutoIntent primarily optimizes single metrics, it considers multiple objectives implicitly:

**Performance vs. Efficiency Trade-offs**

- **Model size**: Smaller models for deployment efficiency  
- **Training time**: Faster models for rapid iteration
- **Inference speed**: Optimized for production latency

**Presets as Multi-Objective Solutions**

AutoIntent provides presets that balance different objectives:

.. code-block:: python

   # Different computational budgets
   pipeline_light = Pipeline.from_preset("classic-light")    # Speed-focused
   pipeline_heavy = Pipeline.from_preset("classic-heavy")    # Performance-focused
   
   # Different model types  
   pipeline_zero_shot = Pipeline.from_preset("zero-shot-transformers")  # No training data

Bayesian Optimization Theory
-----------------------------

**Gaussian Process Surrogate Models**

While TPE uses tree-structured models, the general Bayesian optimization framework uses Gaussian Processes as surrogate models. These are probabilistic models that learn to predict performance based on previous trials, including uncertainty estimates about unexplored regions of the hyperparameter space.

**Exploration vs. Exploitation**

Bayesian optimization balances:

- **Exploitation**: Sampling near known good configurations
- **Exploration**: Sampling in uncertain regions of the space

The acquisition function mathematically encodes this trade-off.

**Convergence Properties**

TPE and related algorithms have theoretical guarantees for convergence to global optima under certain conditions, though practical performance depends on:

- Search space dimensionality
- Function smoothness  
- Available computational budget

Practical Optimization Strategies
----------------------------------

**Budget Allocation**

.. code-block:: python

   hpo_config = HPOConfig(
       sampler="tpe",
       n_trials=50,              # Total optimization budget
       n_startup_trials=10,      # Random initialization
       timeout=3600,             # 1-hour time limit
       n_jobs=4                  # Parallel trials
   )

**Warm Starting**

AutoIntent can resume interrupted optimization. This is the approximate code we use for creating optuna studies:

.. code-block:: python

   # Optimization state is automatically saved
   study = optuna.create_study(
       study_name="intent_classification",
       storage="sqlite:///optuna.db",
       load_if_exists=True
   )

Advanced Topics
---------------

**Meta-Learning**

AutoIntent's presets can be viewed as meta-learning solutions - configurations that work well across diverse datasets based on empirical analysis.

**Neural Architecture Search (NAS)**

While not fully implemented, AutoIntent's modular design supports architecture search within model families (e.g., different CNN configurations).

**Automated Feature Engineering**

AutoIntent's embedding-centric design can be seen as automated feature engineering - the system automatically learns relevant representations through selecting best fitting embedding model.
