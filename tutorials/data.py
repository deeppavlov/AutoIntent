# %% [markdown]
"""
# Working with data
"""

# %%
import importlib.resources as ires

import datasets

from autointent.context.data_handler import Dataset

# %%
datasets.logging.disable_progress_bar() # disable tqdm outputs

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
        ...
    ]
}
```

For a multilabel dataset, the `label` field should be a list of integers representing the corresponding class labels.

### Handling out-of-scope (OOS) samples

To indicate that a sample is out-of-scope (OOS), omit the `label` field from the sample dictionary. For example:

```json
{
    "train": [
        {
            "utterance": "OOS request"
        },
        ...
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
        ...
    ],
    "test": [
        {
            "utterance": "Hi!",
            "label": 0
        },
        ...
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
        ...
    ],
    "intents": [
        {
            "id": 0,
            "name": "greeting",
            "tags": ["conversation_start"],
            "regexp_partial_match": ["\bhello\b"],
            "regexp_full_match": ["^hello$"],
            "description": "User wants to initiate a conversation with a greeting."
        }
        ...
    ]
}
```

- `name`: A human-readable representation of the intent.
- `tags`: Used in multilabel scenarios to predict the most probable class listed in a specific tag.
- `regexp_partial_match` and `regexp_full_match`: Used by the `RegExp` module to predict intents based on provided patterns.
- `description`: Used by the `DescriptionScorer` to calculate scores based on the similarity between an utterance and intent descriptions.

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
For example, you can load the `autointent/datafiles/dstc3-20shot.json` file like this:
"""

# %%
dataset = Dataset.from_json(
    ires.files("autointent.datafiles").joinpath("dstc3-20shot.json"),
)

# %% [markdown]
"""
### Loading a dataset from the Hugging Face Hub

If your dataset on the Hugging Face Hub matches the required format, you can load it directly using its repository ID:
"""

# %%
# dataset = Dataset.from_datasets("<repo_id>")

# %% [markdown]
"""
### Accessing dataset splits
"""

# %%
dataset = Dataset.from_json(
    ires.files("autointent.datafiles").joinpath("banking77.json"),
)

# %% [markdown]
"""
The `Dataset` class organizes your data as a dictionary of splits (`str: datasets.Dataset`).
For example, after initialization, an `oos` key may be added if OOS samples are provided.
In the case of the `banking77` dataset, only the `train` split is available, which you can access as shown below:
"""

# %%
dataset["train"]

# %% [markdown]
"""
### Working with dataset splits

Each split in the `Dataset` class is an instance of [datasets.Dataset](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset),
so you can work with them accordingly.
"""

# %%
dataset["train"][:5]  # get first 5 train samples

# %% [markdown]
"""
### Working with intents

Metadata that you added to intents in your dataset is stored in `intents: list[Intent]` attribute.
"""

# %%
dataset.intents[0] # get intent (id=0)

# %% [markdown]
"""
### Pushing a dataset to the Hugging Face Hub

To share your dataset on the Hugging Face Hub, use the `push_to_hub` method.
Ensure that you are logged in using the `huggingface-cli` tool:
"""

# %%
# dataset.push_to_hub("<repo_id>")
