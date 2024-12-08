# %% [markdown]
"""
# Search Space as YAML

If you want to use default search space, you can skip this tutorial. Here we discuss how to save your custom search space as YAML file in order to use it in CLI for pipeline auto-configuration.

## YAML

YAML (YAML Ain't Markup Language) is a human-readable data serialization standard that is often used for configuration files and data exchange between languages with different data structures. It serves similar purposes as JSON but is much easier to read.

Here's an example YAML file:


```yaml
database:
    host: localhost
    port: 5432
    username: admin
    # this is a comment
    password: secret

counts:
- 10
- 20
- 30

literal_counts: [10, 20, 30]

users:
- name: Alice
    age: 30
    email: alice@example.com
- name: Bob
    age: 25
    email: bob@example.com

settings:
debug: true
timeout: 30
```

Explanation:

- the whole file represents a dictionary with keys ``database``, ``counts``, ``users``, ``settings``, ``debug``, ``timeout``
- ``database`` itself is a dictionary with keys ``host``, ``port``, and so on
- ``counts`` is a list (Python ``[10, 20, 30]``)
- ``literal_counts`` is a list too
- ``users`` is a list of dictionaries

## Example Search Space

```yaml
- node_type: retrieval
  metric: retrieval_hit_rate
  search_space:
    - module_type: vector_db
      k: [10]
      embedder_name:
        - avsolatorio/GIST-small-Embedding-v0
        - infgrad/stella-base-en-v2
- node_type: scoring
  metric: scoring_roc_auc
  search_space:
    - module_type: knn
      k: [1, 3, 5, 10]
      weights: ["uniform", "distance", "closest"]
    - module_type: linear
    - module_type: dnnc
      cross_encoder_name:
        - BAAI/bge-reranker-base
        - cross-encoder/ms-marco-MiniLM-L-6-v2
      k: [1, 3, 5, 10]
- node_type: prediction
  metric: prediction_accuracy
  search_space:
    - module_type: threshold
      thresh: [0.5]
    - module_type: argmax
```
"""