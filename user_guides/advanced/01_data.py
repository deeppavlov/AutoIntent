# %% [markdown]
"""
# Working with data

This chapter is a more detailed version of data chapter from basic user guide about how to manipulate intent classification data with AutoIntent.
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

To create a dataset, you need to provide a training split containing samples with utterances and labels, as shown below:

```json
{
    "train": [
        {
            "utterance": "Hello!",
            "label": 0
        },
        "...",
    ]
}
```

For a multilabel dataset, the `label` field should be a list of integers representing the corresponding class labels.

### Handling out-of-scope samples

To indicate that a sample is out-of-scope (see %mddoclink(rst,concepts)), omit the `label` field from the sample dictionary. For example:

```json
{
    "train": [
        {
            "utterance": "OOS request"
        },
        "...",
    ]
}
```

### Validation and test splits

By default, a portion of the training split will be allocated for validation and testing.
However, you can also specify a test split explicitly:

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

### Adding metadata to intents

You can add metadata to intents in your dataset, such as
regular expressions, intent names, descriptions, or tags, using the `intents` field:

```json
{
    "train": [
        {
            "utterance": "Hello!",
            "label": 0
        },
        "...",
    ],
    "intents": [
        {
            "id": 0,
            "name": "greeting",
            "tags": ["conversation_start"],
            "regexp_partial_match": ["\bhello\b"],
            "regexp_full_match": ["^hello$"],
            "description": "User wants to initiate a conversation with a greeting."
        },
        "...",
    ]
}
```

- `name`: A human-readable representation of the intent.
- `tags`: Used in multilabel scenarios to predict the most probable class listed in a specific %mddoclink(class,schemas,Tag).
- `regexp_partial_match` and `regexp_full_match`: Used by the %mddoclink(class,modules.regexp,RegExp) module to predict intents based on provided patterns.
- `description`: Used by the %mddoclink(class,modules.scoring,DescriptionScorer) to calculate scores based on the similarity between an utterance and intent descriptions.

All fields in the `intents` list are optional except for `id`.
"""

# %% [markdown]
"""
## Loading a dataset

There are three main ways to load your dataset:

1. From a Python dictionary.
2. From a JSON file.
3. Directly from the Hugging Face Hub.
"""

# %% [markdown]
"""
### Creating a dataset from a Python dictionary

One can load data into Python using our %mddoclink(class,,Dataset) object.
"""

# %%
dataset = Dataset.from_dict(
    {
        "train": [
            {
                "utterance": "Please help me with my card. It won't activate.",
                "label": 0,
            },
            {
                "utterance": "I tried but am unable to activate my card.",
                "label": 0,
            },
            {
                "utterance": "I want to open an account for my children.",
                "label": 1,
            },
            {
                "utterance": "How old do you need to be to use the bank's services?",
                "label": 1,
            },
        ],
        "test": [
            {
                "utterance": "I want to start using my card.",
                "label": 0,
            },
            {
                "utterance": "How old do I need to be?",
                "label": 1,
            },
        ],
        "intents": [
            {
                "id": 0,
                "name": "activate_my_card",
            },
            {
                "id": 1,
                "name": "age_limit",
            },
        ],
    },
)

# %% [markdown]
"""
### Loading a dataset from a file

The AutoIntent library includes sample datasets.
"""

# %%
path_to_dataset = ires.files("tests.assets.data").joinpath("clinc_subset.json")
dataset = Dataset.from_json(path_to_dataset)

# %% [markdown]
"""
### Loading a dataset from the Hugging Face Hub

If your dataset on the Hugging Face Hub matches the required format, you can load it directly using its repository ID:
"""

# %%
dataset = Dataset.from_datasets("AutoIntent/clinc150_subset")

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
### Working with intents

Metadata that you added to intents in your dataset is stored in %mddoclink(method,Dataset,intents) attribute.
"""

# %%
dataset.intents[:3]

# %% [markdown]
"""
### Pushing dataset to the Hugging Face Hub

To share your dataset on the Hugging Face Hub, use method %mddoclink(method,Dataset,push_to_hub).
Ensure that you are logged in using the `huggingface-cli` tool:
"""

# %%
# dataset.push_to_hub("<repo_id>")
