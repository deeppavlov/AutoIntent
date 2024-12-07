# %% [markdown]
"""
# Customize Pipeline Auto Configuration
"""

# %%
from autointent import Pipeline

# %% [markdown]
"""
In this tutorial we will walk through different levels of auto configuration process customization.

Let us use small subset of popular `clinc150` dataset for the demonstation.
"""

# %%
from autointent import Dataset

dataset = Dataset.from_datasets("AutoIntent/clinc150_subset")
dataset

# %%
dataset["train"][0]


# %% [markdown]
"""
## Search Space

AutoIntent provides default search spaces for multi-label and single-label classification problems. One can utilize them by constructing %mddoclink(class,,Pipeline) with factory %mddoclink(method,Pipeline,default_optimizer):
"""

# %%
multiclass_pipeline = Pipeline.default_optimizer(multilabel=False)
multilabel_pipeline = Pipeline.default_optimizer(multilabel=True)

# %% [markdown]
"""
One can explore its contents:
"""

# %%
from autointent.utils import load_default_search_space

search_space = load_default_search_space(multilabel=True)

# %% [markdown]
"""
Search space is allowed to customize:
"""

# %%
search_space[1]["search_space"][0]["k"] = [1, 3]
custom_pipeline = Pipeline.from_search_space(search_space)

# %% [markdown]
"""
See tutorial %mddoclink(tutorial,auto_configuration.search_space_configuration) on how the search space is structured.
"""

# %% [markdown]
"""
## Embedder Settings

%mddoclink(class,,Embedder) is one of the key components of AutoIntent. It affects both the quality of the resulting classifier and the efficiency of the auto configuration process.

To select embedding models for your optimization, you need to customize search (%mddoclink(tutorial,auto_configuration.search_space_configuration)). Here, we will observe settings affecting efficiency.

Several options are customizable. Defaults are the following:
"""

# %%
from autointent.configs import EmbedderConfig

embedder_config = EmbedderConfig(
    batch_size=32,
    max_length=None,
    use_cache=False,
)

# %% [markdown]
"""
To set selected settings, method %mddoclink(method,Pipeline,set_config) is provided:
"""

# %%
custom_pipeline.set_config(embedder_config)

# %% [markdown]
"""
## Vector Index Settings

%mddoclink(class,context.vector_index_client,VectorIndex) is one of the key utilities of AutoIntent. It affects both the quality of the resulting classifier and the efficiency of the auto configuration process.
"""
