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
