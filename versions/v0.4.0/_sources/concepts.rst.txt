============
Key Concepts
============

This page introduces the fundamental concepts that underpin AutoIntent's design and functionality. Understanding these concepts will help you effectively use the framework and make informed decisions about your text classification projects.

.. _concepts-pipeline:

Three-Stage Pipeline Architecture
=================================

AutoIntent organizes text classification into a modular three-stage pipeline, providing clear separation of concerns and flexibility in optimization:

**🔤 Embedding Stage**
   Transforms raw text into dense vector representations using pre-trained transformer models. This stage handles the computationally intensive text encoding and can be optimized independently from downstream classification tasks.

**📊 Scoring Stage**
   Processes embeddings to predict class probabilities. This stage supports diverse approaches from classical machine learning (KNN, logistic regression) to deep learning models (BERT fine-tuning, CNNs). All models operate on pre-computed embeddings for efficiency.

**⚖️ Decision Stage**
   Converts predicted probabilities into final classifications by applying thresholds and decision rules. This stage is crucial for multi-label classification and out-of-scope detection scenarios.

This modular design enables efficient experimentation, allows reusing expensive embedding computations across different models, and supports deployment on CPU-only systems.

.. _concepts-automl:

AutoML Optimization Strategy
============================

AutoIntent employs a hierarchical optimization approach that balances exploration with computational efficiency:

**🔧 Module-Level Optimization**
   Components are optimized sequentially: embedding → scoring → decision. Each stage builds upon the best model from the previous stage, creating a cohesive pipeline while preventing combinatorial explosion.

**🤖 Model-Level Optimization**
   Within each module, both model architectures and hyperparameters are jointly optimized using Optuna's Tree-structured Parzen Estimators and random sampling.

**🗺️ Search Space Configuration**
   Optimization behavior is controlled through dictionary-like search spaces that define:
   
   - Available model types and their hyperparameter ranges
   - Optimization budget and resource constraints  
   - Cross-validation and evaluation strategies

.. _concepts-embedding-centric:

Embedding-Centric Design
========================

AutoIntent's architecture centers around transformer-based text embeddings, providing several key advantages:

**⚡ Pre-computed Embeddings**
   Text is encoded once and reused across all scoring models, dramatically reducing computational overhead during hyperparameter optimization and enabling efficient experimentation.

**🤗 Model Repository Integration**
   Seamless access to thousands of pre-trained models from Hugging Face Hub, with intelligent selection strategies based on retrieval metrics or downstream task performance.

**🚀 Deployment Flexibility**
   Separation of embedding generation from classification enables deploying lightweight classifiers on resource-constrained systems while leveraging powerful transformer representations.

.. _concepts-multiclass-multilabel:

Multi- vs. Single-label classification
======================================

AutoIntent supports various classification scenarios through its flexible decision module:

**🏷️ Multi-Class Classification**
   Each input gets assigned to exactly one category - like sorting emails into "Spam", "Work", or "Personal" folders. Common examples include sentiment analysis (positive/negative/neutral) or determining user intent where each message has a single purpose. The model picks the single best match from all possible categories.

**🔖 Multi-Label Classification** 
   Each input can belong to multiple categories at once - like tagging a news article as both "Politics" and "Economics". Essential for scenarios like multi-intent messages ("book a flight and check weather"), content tagging, or any situation where multiple labels can apply simultaneously. The model almost independently decides whether each possible category fits or not.


.. _concepts-oos:

Out-of-Scope Detection
======================

A critical capability for production text classification systems, especially in conversational AI:

**📏 Confidence Thresholding**
   Uses predicted probability scores to identify inputs that don't belong to any known class. Threshold values can be tuned automatically to balance precision and recall.

**🔗 Integration with Multi-Label**
   OOS detection works seamlessly with multi-label scenarios, enabling detection of completely unknown inputs vs. partial matches to known classes.

**🧭 Split handling**
   When splits contain OOS samples (``label is None``), the data handler keeps scoring stages on in-domain rows only: in hold-out mode it can duplicate affected splits into ``{split}_0`` (OOS removed for scoring) and ``{split}_1`` (full data for decision) when ``separation_ratio`` is not configured, and cross-validation similarly drops OOS from training folds used while scoring. Before fitting, you can validate whether your data supports splitting with :py:func:`autointent.context.data_handler.check_split_readiness`.

.. _concepts-presets:

Optimization Presets
====================

AutoIntent provides predefined optimization strategies that balance quality, speed, and resource consumption:

**⚡ Zero-Shot Presets**
   Leverage class descriptions and large language models for classification without training data. Ideal for rapid prototyping and cold-start scenarios.

**📈 Classic Presets**
   Focus on traditional ML approaches (KNN, linear models, tree-based methods) operating on transformer embeddings. Offer excellent balance of performance and efficiency.

**🧠 Neural Network Presets**
   Include deep learning approaches like CNN, RNN, and transformer fine-tuning. Provide highest potential performance at increased computational cost.

**🪜 Computational Tiers**
   Each preset family offers light, medium, and heavy variants that trade optimization time for potential performance improvements.

.. _concepts-modularity:

Modular Architecture
====================

AutoIntent's design emphasizes modularity and extensibility:

**🧩 Plugin Architecture**
   Each component (embedding models, scoring methods, decision strategies) implements a common interface, enabling easy addition of new approaches without modifying core framework code.

**⚙️ Configuration-Driven**
   All aspects of optimization can be controlled through declarative configuration files, supporting reproducible experiments and easy sharing of optimization strategies.

**🔧 Extensibility**
   Framework can be extended with custom embedding models, scoring algorithms, and decision strategies while maintaining compatibility with the AutoML optimization pipeline.

This modular design ensures that AutoIntent can evolve with advances in NLP research while maintaining stability and backward compatibility for existing users.

Vector Index Lifecycle
======================

All vector-index backends follow one contract: the **live index a module fits into is transient training scratch**, owned by whichever instance is currently fitting (each ``fit()`` replaces its contents); **durability comes from** ``dump()`` **artifacts**. During hyperparameter search only the best module of each node is dumped, and trials keep rewriting the scratch index after the best one — so never serve from the live index; serve from a loaded dump.

How ``dump()`` achieves durability differs by backend:

- **Faiss** writes a self-contained directory (documents + binary index).
- **OpenSearch** performs *copy-on-dump*: the live index is copied server-side (``_reindex``) into a write-blocked *generation* index named ``{base}-best-{id}``, and the dump directory records it in ``remote_manifest.json``. The corpus never leaves the cluster. ``Pipeline.load`` binds to that immutable generation, verifies its identity, and never writes; it fails loudly if the generation was deleted or recreated.

Notes specific to the OpenSearch backend:

- **Cleanup.** Delete dumps with :func:`autointent.remove_module_dump` — it removes the referenced generation index from the cluster along with the directory. A plain ``rm -rf`` strands the generation (recoverable: generations are pattern-named ``*-best-*``, so ops can also enforce an ISM age policy). AutoIntent's optimizer uses this helper automatically when a new best trial replaces the previous one, so during optimization the steady-state cluster footprint is the live scratch index plus one generation per node type; a later ``Pipeline.dump()`` re-dump of an already-optimized pipeline adds one more generation per node, which lingers until the corresponding dump directory is deleted.
- **Naming.** Avoid naming your own live indices with a ``-best-`` infix — ``{base}-best-*`` is the namespace AutoIntent uses for generations and the serving alias.
- **Portability.** A dumped OpenSearch pipeline is portable only as far as the cluster: copying the dump directory to an environment that cannot reach the same cluster carries a dangling reference. This is inherent to keeping the corpus in the engine.
- **Querying an existing collection.** Do not point ``OpenSearchConfig.index_name`` at a collection you want to keep — ``fit()`` clears it (fit-replaces). To serve an existing corpus read-only, fit and dump once, then use ``Pipeline.load`` + ``predict``: loaded pipelines only ever read their immutable generation.
- **Parallel trials.** Hyperparameter search with ``n_jobs > 1`` would share one live scratch index across concurrently fitting trials; keep ``n_jobs = 1`` when using the OpenSearch backend.
