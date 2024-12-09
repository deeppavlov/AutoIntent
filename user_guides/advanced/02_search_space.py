# %% [markdown]
"""
# Search Space Configuration

In this guide, you will learn how to configure a custom hyperparameter search space.
"""

# %% [markdown]
"""
## Python API

> Before reading this guide, we recommend familiarizing yourself with the sections %mddoclink(rst,concepts) and %mddoclink(rst,learn.optimization).
"""

# %% [markdown]
"""
### Optimization Module


To set up the optimization module, you need to create the following dictionary:
"""

# %%
knn_module = {
    "module_name": "knn",
    "k": [1, 5, 10, 50],
    "embedder_name": ["avsolatorio/GIST-small-Embedding-v0", "infgrad/stella-base-en-v2"],
}

# %% [markdown]
"""
The ``module_name`` field specifies the name of the module. You can find the names, for example, in...

TODO: _Add docs for all available modules._

All fields except ``module_name`` are lists that define the search space for each hyperparameter (see %mddoclink(class,modules.scoring,KNNScorer)). If you omit them, the default set of hyperparameters will be used:
"""

# %%

linear_module = {"module_name": "linear"}

# %% [markdown]
"""
See docs %mddoclink(class,modules.scoring,LinearScorer).
"""

# %% [markdown]
"""
### Optimization Node

To set up the optimization node, you need to create a list of modules and specify the metric for optimization:
"""

# %%
scoring_node = {
    "node_type": "scoring",
    "metric_name": "scoring_roc_auc",
    "search_space": [
        knn_module,
        linear_module,
    ],
}

# %% [markdown]
"""
### Search Space

The search space for the entire pipeline looks approximately like this:
"""

# %%
search_space = [
    {
        "node_type": "retrieval",
        "metric": "retrieval_hit_rate",
        "search_space": [
            {
                "module_name": "vector_db",
                "k": [10],
                "embedder_name": ["avsolatorio/GIST-small-Embedding-v0", "infgrad/stella-base-en-v2"],
            }
        ],
    },
    {
        "node_type": "scoring",
        "metric": "scoring_roc_auc",
        "search_space": [
            {"module_name": "knn", "k": [1, 3, 5, 10], "weights": ["uniform", "distance", "closest"]},
            {"module_name": "linear"},
            {
                "module_name": "dnnc",
                "cross_encoder_name": ["BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
                "k": [1, 3, 5, 10],
            },
        ],
    },
    {
        "node_type": "decision",
        "metric": "decision_accuracy",
        "search_space": [{"module_name": "threshold", "thresh": [0.5]}, {"module_name": "argmax"}],
    },
]

# %% [markdown]
"""
### Load Data

Let us use small subset of popular `clinc150` dataset:
"""

# %%

from autointent import Dataset

dataset = Dataset.from_hub("AutoIntent/clinc150_subset")

# %% [markdown]
"""
### Start Auto Configuration
"""

# %%
from autointent import Pipeline

pipeline_optimizer = Pipeline.from_search_space(search_space)
pipeline_optimizer.fit(dataset)

# %% [markdown]
"""
## See Also

- [Modules API reference](../autoapi/autointent/modules/index.rst) to get familiar with modules to include into search space
"""
