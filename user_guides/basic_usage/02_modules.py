# %% [markdown]
"""
# Modules

In this chapter you will get familiar with modules and how to use them for intent classification.

Modules are the basic units in our library. They perform core operations such as predicting probabilities and constructing final set of predicted labels.

## Modules Types

There are two main module types in AutoIntent:
- **Scoring modules.** These modules perform probabilities prediction, i.e. they take an utterance as input and output a vector of probabilities.
- **Prediction modules.** These modules take vector of probabilities and output set of labels. Prediction modules are important to support multi-label classification and out-of-domain utterances detection.

## Initialize Module

Firstly, you need to initialize module:
"""

# %%
from autointent.modules.scoring import KNNScorer

scorer = KNNScorer(
    embedder_name="sergeyzh/rubert-tiny-turbo",
    k=5,
)

# %% [markdown]
"""
At this moment, you do two things:
- **Set hyperparameters**. Refer to [Modules API Reference](../autoapi/autointent/modules/index.rst) to see all possible hyperparameters and their default values.
- **Configure infrastructure**. You are allowed to
    - choose CUDA device (`embedder_device`)
    - customize embedder batch size (`batch_size`) and truncation length (`embedder_max_length`)
    - location where to save module's assets (`db_dir`)

## Load Data

Secondly, you need to load training data (see previous chapter for detailed explanation of what happens):
"""

# %%
from autointent import Dataset

dataset = Dataset.from_datasets("AutoIntent/clinc150_subset")

# %% [markdown]
"""
## Fit Module
"""

# %%
scorer.fit(dataset["train"]["utterance"], dataset["train"]["label"])

# %% [markdown]
"""
## Inference

After fitting, module is ready for using at inference:
"""

# %%
scorer.predict(["hello world!"])

# %% [markdown]
"""
## Dump and Load

We provide functionality to save and restore module. To save, just provide a path to a directory:
"""

# %%
from pathlib import Path

pathdir = Path("my_dumps/knnscorer_clinc150")
pathdir.mkdir(parents=True)
scorer.dump(pathdir)

# %% [markdown]
"""
To restore, initialize module with the same hyperparams and use load method:
"""

# %%
loaded_scorer = KNNScorer(
    embedder_name="sergeyzh/rubert-tiny-turbo",
    k=5,
)
loaded_scorer.load(pathdir)
loaded_scorer.predict(["hello world!"])

# %% [markdown]
"""
## Rich Output

Some scoring modules support rich output as a result of prediction. It can be useful for inspecting how your classifier work and for debugging as it contains intrinsic information such as retrieved candidates. Example:
"""

# %%
loaded_scorer.predict_with_metadata(["hello world!"])

# %% [markdown]
"""
## That's all!
"""

# %%
# [you didn't see it]
import shutil

shutil.rmtree(pathdir.parent)

for file in Path.cwd().glob("vector_db*"):
    shutil.rmtree(file)
