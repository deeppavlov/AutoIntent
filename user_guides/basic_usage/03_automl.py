# %% [markdown]
"""
# AutoML
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
## Default Transformers

One can specify what embedding model and cross-encoder model want to use along with default settings:
"""

# %%
from autointent.configs import EmbedderConfig, CrossEncoderConfig

custom_pipeline.set_config(EmbedderConfig(model_name="prajjwal1/bert-tiny", device="cpu"))
custom_pipeline.set_config(CrossEncoderConfig(model_name="cross-encoder/ms-marco-MiniLM-L2-v2", max_length=8))

# %% [markdown]
"""
See the docs for %mddoclink(class,configs,EmbedderConfig) and %mddoclink(class,configs,CrossEncoderConfig) for options available to customize.
"""

# %% [markdown]
"""
## Cross-Validation vs Hold-Out Validation

If you have lots of training and evaluation data, you can use default hold-out validation strategy. If not, you can choose cross-validation and spend a little more time but utilize the full amount of available data for better hyperparameter tuning.

This behavior is controlled with %mddoclink(class,configs,DataConfig):
"""

# %%
from autointent.configs import DataConfig
custom_pipeline.set_config(DataConfig(scheme="cv", n_folds=3))

# %% [markdown]
"""
See the docs for %mddoclink(class,configs,DataConfig) for other options available to customize.
"""

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
context = custom_pipeline.fit(dataset)

# inference on-the-fly
custom_pipeline.predict(["hello world!"])

# %% [markdown]
"""
## Dump Results

One can save all results of auto-configuration process to file system (to ``LoggingConfig.dirpath``):
"""

# %%
context.dump()

# %% [markdown]
"""
Or one can dump only the configured pipeline to any desired location (by default ``LoggingConfig.dirpath``):
"""

# %%
custom_pipeline.dump()

# %% [markdown]
"""
## Load Pipeline for Inference
"""

# %%
loaded_pipe = Pipeline.load(logging_config.dirpath)

# %% [markdown]
"""
Since this notebook is launched automatically while building the docs, we will clean the space if you don't mind :)
"""

# %%
import shutil

shutil.rmtree(logging_config.dirpath)
