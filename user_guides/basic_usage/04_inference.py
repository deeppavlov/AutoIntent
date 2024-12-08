# %% [markdown]
"""
# Inference Pipeline

After you configured optimal pipeline with AutoIntent, you probably want to test its power on some new data! There are several options:
- use it right after optimization
- save to file system and then load

## Right After

Here's the basic example:
"""

# %%
from autointent import Pipeline, Dataset

dataset = Dataset.from_datasets("AutoIntent/clinc150_subset")
pipeline = Pipeline.default_optimizer(multilabel=False)
context = pipeline.fit(dataset)
pipeline.predict(["hello, world!"])

# %% [markdown]
"""
There are several caveats.

1. **Save vector databse.**

When customizing configuration of pipeline optimization, you need to ensure that the option `save_db` of %mddoclink(class,configs,VectorIndexConfig) is set to `True`:
"""
# %%
from autointent.configs import VectorIndexConfig

# isn't compatible with "right-after-optimization" inference
vector_index_config = VectorIndexConfig(save_db=False)

# %% [markdown]
"""
2. **RAM usage.**

You can optimize RAM usage by saving all modules to file system. Just set the following options:
"""

# %%
from autointent.configs import LoggingConfig

logging_config = LoggingConfig(dump_modules=True, clear_ram=True)

# %% [markdown]
"""
## Load from File System

Firstly, your auto-configuration run should dump modules into file system:
"""

# %%
from autointent import Pipeline, Dataset
from autointent.configs import LoggingConfig, VectorIndexConfig
from pathlib import Path

dataset = Dataset.from_datasets("AutoIntent/clinc150_subset")
pipeline = Pipeline.default_optimizer(multilabel=False)
dump_dir = Path("my_dumps")
pipeline.set_config(LoggingConfig(dump_dir=dump_dir, dump_modules=True, clear_ram=True))
pipeline.set_config(VectorIndexConfig(save_db=True))

# %% [markdown]
"""
Secondly, after optimization finished, you need to save the auto-configuration results to file system:
"""

# %%
context = pipeline.fit(dataset)
context.dump()

# %% [markdown]
"""
This command saves all results to the run's directory:
"""

# %%
run_directory = context.logging_config.dirpath
run_directory

# %% [markdown]
"""
After that, you can load pipeline for inference:
"""

# %%
loaded_pipeline = Pipeline.load(run_directory)
loaded_pipeline.predict(["hello, world!"])

# %%
# [you didn't see it]
import shutil
shutil.rmtree(dump_dir)

for file in Path.glob("./vector_db*"):
    shutil.rmtree(file)