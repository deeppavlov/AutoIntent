# %% [markdown]
"""
# Pipeline Auto Configuration (AutoML)
"""

# %%
from autointent import Pipeline

# %% [markdown]
"""
In this tutorial we will walk through pipeline auto configuration process.

Let us use small subset of popular `clinc150` dataset for the demonstation.
"""

# %%
from autointent import Dataset

dataset = Dataset.from_hub("AutoIntent/clinc150_subset")
dataset

# %%
dataset["train_0"][0]


# %% [markdown]
"""
## Search Space

AutoIntent provides default search spaces. One can utilize them by constructing %mddoclink(class,,Pipeline) with factory %mddoclink(method,Pipeline,from_preset):
"""

# %%
pipeline = Pipeline.from_preset("light_extra")

# %% [markdown]
"""
One can explore its contents:
"""

# %%
from pprint import pprint

from autointent.utils import load_preset

preset = load_preset("light_extra")
pprint(preset)

# %% [markdown]
"""
Search space is allowed to customize:
"""

# %%
preset["search_space"][0]["search_space"][0]["k"] = [1, 3]
custom_pipeline = Pipeline.from_optimization_config(preset)

# %% [markdown]
"""
See tutorial %mddoclink(notebook,advanced.02_search_space_configuration) on how the search space is structured.
"""

# %% [markdown]
"""
## Logging Settings

The important thing is what assets you want to save during the pipeline auto-configuration process. You can control it with %mddoclink(class,configs,LoggingConfig):
"""

# %%
from pathlib import Path
from autointent.configs import LoggingConfig

logging_config = LoggingConfig(project_dir=Path.cwd() / "runs", dump_modules=False, clear_ram=False)
custom_pipeline.set_config(logging_config)

# %% [markdown]
"""
## Complete Example
"""

# %%
from autointent import Dataset, Pipeline
from autointent.configs import LoggingConfig
from autointent.utils import load_preset

# load data
dataset = Dataset.from_hub("AutoIntent/clinc150_subset")

# customize search space
preset = load_preset("light_extra")

# make pipeline
custom_pipeline = Pipeline.from_optimization_config(preset)

# custom settings
logging_config = LoggingConfig()

custom_pipeline.set_config(logging_config)

# start auto-configuration
custom_pipeline.fit(dataset)

# inference
custom_pipeline.predict(["hello world!"])
