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
from pprint import pprint

search_space = load_default_search_space(multilabel=True)
pprint(search_space)

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

Several options are customizable via %mddoclink(class,configs,EmbedderConfig). Defaults are the following:
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

%mddoclink(class,context.vector_index_client,VectorIndex) is one of the key utilities of AutoIntent. During the auto-configuration process, lots of retrieval is used. By modifying %mddoclink(class,configs,VectorIndexConfig) you can select whether to save built vector index into file system and where to save it.

Default options are the following:
"""

# %%
from autointent.configs import VectorIndexConfig

vector_index_config = VectorIndexConfig(
    db_dir=None,
    save_db=False
)

# %% [markdown]
"""
- `db_dir=None` tells AutoIntent to store intermediate files in a current working directory
- `save_db=False` tells AutoIntent to clear all the files after auto configuration is finished

These settings can be applied in a familiar way:
"""

# %%
custom_pipeline.set_config(vector_index_config)

# %% [markdown]
"""
## Logging Settings

The important thing is what assets you want to save during the pipeline auto-configuration process. You can control it with %mddoclink(class,configs,LoggingConfig). Default settings are the following:
"""

# %%
from autointent.configs import LoggingConfig

logging_config = LoggingConfig(
    run_name=None,
    dirpath=None,
    dump_dir=None,
    dump_modules=False,
    clear_ram=False
)
custom_pipeline.set_config(logging_config)

# %% [markdown]
"""
## Complete Example
"""

# %%
from autointent import Pipeline, Dataset
from autointent.utils import load_default_search_space
from autointent.configs import LoggingConfig, VectorIndexConfig, EmbedderConfig

# load data
dataset = Dataset.from_datasets("AutoIntent/clinc150_subset")

# customize search space
search_space = load_default_search_space(multilabel=False)

# make pipeline
custom_pipeline = Pipeline.from_search_space(search_space)

# custom settings
embedder_config = EmbedderConfig()
vector_index_config = VectorIndexConfig()
logging_config = LoggingConfig()

custom_pipeline.set_config(embedder_config)
custom_pipeline.set_config(vector_index_config)
custom_pipeline.set_config(logging_config)

# start auto-configuration
custom_pipeline.fit(dataset)

# inference
custom_pipeline.predict(["hello world!"])

