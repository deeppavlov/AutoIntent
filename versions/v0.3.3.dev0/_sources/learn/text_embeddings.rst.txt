Text Embeddings and Representation Learning
===========================================

In this section, you will learn about the theoretical foundations of text embeddings and how AutoIntent leverages them for efficient intent classification.

What are Text Embeddings?
--------------------------

Text embeddings are dense vector representations of text that capture semantic meaning in a continuous vector space. Unlike traditional bag-of-words approaches that treat words as discrete tokens, embeddings map text to points in a high-dimensional space where semantically similar texts are located close to each other.

**Mathematical Foundation**

An embedding function :math:`f: \mathcal{T} \rightarrow \mathbb{R}^d` maps text :math:`t \in \mathcal{T}` to a dense vector :math:`\mathbf{e} \in \mathbb{R}^d`, where :math:`d` is the embedding dimension (typically 384, 768, or 1024). The key property is that semantic similarity in text space translates to geometric proximity in embedding space:

.. math::
   \text{semantic_similarity}(t_1, t_2) \approx \cos(\mathbf{e}_1, \mathbf{e}_2)

where :math:`\cos(\mathbf{e}_1, \mathbf{e}_2) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{||\mathbf{e}_1|| \cdot ||\mathbf{e}_2||}`

Transformer-Based Embeddings
-----------------------------

AutoIntent primarily uses transformer-based embedding models, which have revolutionized natural language processing through their attention mechanisms and contextual representations.

**Sentence Transformers**

The library leverages the `sentence-transformers <https://www.sbert.net/>`_ framework, which provides pre-trained models specifically optimized for semantic similarity tasks. These models are fine-tuned versions of BERT, RoBERTa, or other transformer architectures that produce high-quality sentence-level embeddings.

**Key Advantages:**

1. **Contextual Understanding**: Unlike word2vec or GloVe, transformer embeddings understand context. The word "bank" will have different representations in "river bank" vs. "money bank."

2. **Cross-lingual Capabilities**: Many models support multiple languages, crucial for dialog systems serving diverse users.

3. **Task Adaptation**: Models can be fine-tuned for specific domains or similarity tasks.

**Model Types in AutoIntent:**

- **Bi-encoders**: Encode texts independently, enabling efficient pre-computation and caching
- **Cross-encoders**: Process text pairs jointly for higher accuracy but at computational cost


Task-Specific Prompting
-----------------------

AutoIntent supports task-specific prompts to optimize embedding quality for different use cases.

Different tasks may benefit from different prompting strategies:

.. code-block:: python

   # Query prompt for search
   query_embeddings = embedder.embed(queries, TaskTypeEnum.query)
   
   # Passage prompt for documents
   doc_embeddings = embedder.embed(documents, TaskTypeEnum.passage)
   
   # Classification prompt for intents
   intent_embeddings = embedder.embed(utterances, TaskTypeEnum.classification)

Embedding Quality and Evaluation
---------------------------------

AutoIntent evaluates embedding quality using retrieval metrics:

- **NDCG** (Normalized Discounted Cumulative Gain)
- **Hit Rate** (Proportion of relevant items in top-k results)
- **Precision@k** and **Recall@k**


Practical Applications in Dialog Systems
-----------------------------------------

**Intent Classification Pipeline**

1. **User utterance**: "I want to book a flight to Paris"
2. **Embedding**: Convert to 768-dimensional vector
3. **Similarity search**: Find nearest training examples
4. **Classification**: Use embedding-based classifier (KNN, linear, etc.)
5. **Decision**: Apply confidence thresholds for final prediction

**Zero-Shot Classification**

Using intent descriptions for classification without training data:

.. code-block:: python

   from autointent.modules.scoring import BiEncoderDescriptionScorer
   
   scorer = BiEncoderDescriptionScorer()
   
   # Intent descriptions instead of training data
   descriptions = [
       "User wants to book a flight",
       "User wants to cancel a reservation",
       "User asks about flight status"
   ]
   
   scorer.fit([], [], descriptions)
   predictions = scorer.predict(["I want to fly to London"])

**Few-Shot Learning**

Embeddings excel in few-shot scenarios where limited training data is available. AutoIntent's k-NN based methods are particularly effective.