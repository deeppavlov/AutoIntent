# %% [markdown]
"""
# Working with data

In this chapter you will learn how to manipulate intent classification data with AutoIntent.
"""

# %%
import importlib.resources as ires

import datasets

from autointent import Dataset

# %%
datasets.logging.disable_progress_bar()  # disable tqdm outputs

# %% [markdown]
"""
## Creating a dataset

The first thing you need to think about is your data. You need to collect a set of labeled utterances and save it as JSON file with the following schema:

```json
{
    "train": [
        {
            "utterance": "Hello!",
            "label": 0
        },
        "...",
    ],
    "test": [
        {
            "utterance": "Hi!",
            "label": 0
        },
        "...",
    ]
}
```

Note:
- For a multilabel dataset, the `label` field should be a list of integers representing the corresponding class labels.
- Test split is optional. By default, a portion of the training split will be allocated for testing.
"""

# %% [markdown]
"""
## Loading a dataset

After you converted your labeled data into JSON, you can load it into AutoIntent as %mddoclink(class,,Dataset). We will load sample dataset that is provided by AutoIntent library to demonstrate this functionality.
"""

# %%
path_to_dataset = ires.files("autointent._datafiles").joinpath("dstc3-20shot.json")
dataset = Dataset.from_json(path_to_dataset)

# %% [markdown]
"""
Note: to load your data, just change `path_to_dataset` variable.
"""

# %% [markdown]
"""
### Accessing dataset splits

The %mddoclink(class,,Dataset) class organizes your data as a dictionary of [datasets.Dataset](https://huggingface.co/docs/datasets/v2.1.0/en/package_reference/main_classes#datasets.Dataset).
For example, after initialization, an `oos` key may be added if OOS samples are provided.
"""

# %%
dataset["train"]

# %% [markdown]
"""
### Working with dataset splits

Each split in the %mddoclink(class,,Dataset) class is an instance of [datasets.Dataset](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset),
so you can work with them accordingly.
"""

# %%
dataset["train"][:5]  # get first 5 train samples

# %% [markdown]
"""
### Save Dataset

To share your dataset on the Hugging Face Hub, use method %mddoclink(method,Dataset,push_to_hub).
"""

# %%
# dataset.push_to_hub("<repo_id>")

# %% [markdown]
"""
Note: ensure that you are logged in using `huggingface-cli`.
"""

# %% [markdown]
"""
## See Also

- Next chapter of the user guide "Using modules": %mddoclink(tutorial,python_api.02_modules)
"""
