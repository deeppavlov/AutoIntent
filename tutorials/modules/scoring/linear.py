# %% [markdown]
"""
# Linear Scoring
"""

# %%
import importlib.resources as ires

from autointent.context._utils import load_data
from autointent.context.data_handler import DataHandler
from autointent.modules import LinearScorer

# %% [markdown]
"""
Load your dataset in our format:
"""

# %%
dataset_path = ires.files("tests.assets.data").joinpath("clinc_subset.json")
dataset = load_data(dataset_path)

# %% [markdown]
"""
Split your data into train and dev sets.
"""

# %%

data_handler = DataHandler(dataset)

# %% [markdown]
"""
Initialize scoring module. It will take text and output probabilities for each class.
"""

# %%
scorer = LinearScorer(embedder_name="sergeyzh/rubert-tiny-turbo", device="cpu", n_jobs=1)


# %% [markdown]
"""
Train model:
"""

# %%

scorer.fit(data_handler.train_utterances, data_handler.train_labels)

# %% [markdown]
"""
We will use the following utterances for testing:
"""

# %%
test_data = [
    "why is there a hold on my american saving bank account",
    "i am nost sure why my account is blocked",
    "why is there a hold on my capital one checking account",
    "i think my account is blocked but i do not know the reason",
    "can you tell me why is my bank account frozen",
]

# %% [markdown]
"""
Predict:
"""

# %%

scorer.predict(test_data)

# %% [markdown]
"""
In general, you can obtain rich information about inference. In case of linear scorer it's empty
"""

# %%

predictions, metadata = scorer.predict_with_metadata(test_data)
assert metadata is None  # noqa: S101
