# %% [markdown]
"""
# Data

In this chapter you will learn how to work with intent classification data in AutoIntent. We'll cover creating datasets, loading data from different sources, and manipulating your data for optimal results.
"""

# %%
import datasets

from autointent import Dataset

# %%
datasets.logging.disable_progress_bar()  # disable tqdm outputs

# %% [markdown]
"""
## Creating your first dataset

The easiest way to get started is by creating a dataset from a Python dictionary. Let's start with a simple banking intent classification example:
"""

# %%
# Create a simple intent classification dataset
data = {
    "train": [
        {"utterance": "What is my account balance?", "label": 0},
        {"utterance": "Check my current balance", "label": 0},
        {"utterance": "Show me my account details", "label": 0},
        {"utterance": "I want to transfer money", "label": 1},
        {"utterance": "How do I send funds to someone?", "label": 1},
        {"utterance": "Make a payment to my friend", "label": 1},
        {"utterance": "Cancel my last transaction", "label": 2},
        {"utterance": "Reverse the payment I just made", "label": 2},
        {"utterance": "Stop this transfer", "label": 2},
    ],
    "validation": [
        {"utterance": "Display my balance", "label": 0},
        {"utterance": "Send money to John", "label": 1},
        {"utterance": "Cancel the last payment", "label": 2},
    ],
    "test": [
        {"utterance": "How much money is in my account?", "label": 0},
        {"utterance": "Transfer funds to my savings", "label": 1},
        {"utterance": "Undo my recent payment", "label": 2},
    ],
}

# Load the data into AutoIntent
dataset = Dataset.from_dict(data)
print(f"Dataset created with {len(dataset['train'])} training samples")

# %% [markdown]
"""
This creates a dataset with three intent classes:
- **0**: Balance inquiries
- **1**: Money transfers
- **2**: Transaction cancellations

**Important notes about data splits:**
- **Test split**: Highly recommended as a frozen evaluation set that's never used during training
- **Validation split**: Optional - if not provided, AutoIntent will automatically split your training data
- **Training split**: Required - this is where your model learns from
"""

# %% [markdown]
"""
## Understanding the data format

AutoIntent expects your data in a specific format. Here are the key requirements:

### Single-label classification
For most intent classification tasks, each utterance belongs to exactly one class:

```python
{
    "train": [
        {"utterance": "Hello!", "label": 0},
        {"utterance": "Book a flight", "label": 1},
        {"utterance": "What's the weather?", "label": 2}
    ],
    "test": [  # Recommended: frozen test set
        {"utterance": "Hi there!", "label": 0}
    ]
    # validation split is optional - AutoIntent will create one if needed
}
```

### Multi-label classification
For tasks where utterances can belong to multiple classes, use a list of labels:

```python
{
    "train": [
        {"utterance": "Book urgent flight to Paris", "label": [1, 0, 1]},  # booking=1, weather=0, urgent=1
        {"utterance": "What's the weather like?", "label": [0, 1, 0]}
    ],
    "test": [
        {"utterance": "Emergency flight booking", "label": [1, 0, 1]}
    ]
}
```
"""

# %% [markdown]
"""
## Loading data from different sources

AutoIntent supports multiple ways to load your data:
"""

# %% [markdown]
"""
### From a dictionary (recommended for getting started)
Perfect when you have your data ready in Python:
"""

# %%
# Example with a complete dataset including all splits
banking_data = {
    "train": [
        {"utterance": "What is my account balance?", "label": 0},
        {"utterance": "Check my savings balance", "label": 0},
        {"utterance": "How much money do I have?", "label": 0},
        {"utterance": "Transfer $100 to savings", "label": 1},
        {"utterance": "Send money to my friend", "label": 1},
        {"utterance": "Make a payment", "label": 1},
        {"utterance": "Cancel my last payment", "label": 2},
        {"utterance": "Stop this transaction", "label": 2},
        {"utterance": "Reverse my transfer", "label": 2},
    ],
    "validation": [
        {"utterance": "Display my balance", "label": 0},
        {"utterance": "Send $50 to John", "label": 1},
        {"utterance": "Stop my last transaction", "label": 2},
    ],
    "test": [
        {"utterance": "Show me my current balance", "label": 0},
        {"utterance": "I want to transfer funds", "label": 1},
        {"utterance": "Cancel this payment", "label": 2},
    ],
}

dataset_from_dict = Dataset.from_dict(banking_data)
print("✅ Dataset loaded from dictionary")
print(f"Splits: {list(dataset_from_dict.keys())}")

# %% [markdown]
"""
### From a JSON file
When you have your data saved as a JSON file with the same structure:
"""

# %%
# Example: dataset_from_json = Dataset.from_json("/path/to/your/data.json")

# %% [markdown]
"""
### From Hugging Face Hub
For loading public datasets or sharing your own:
"""

# %%
# Load a sample dataset from HuggingFace Hub
dataset_from_hub = Dataset.from_hub("DeepPavlov/banking77")
print("✅ Dataset loaded from Hugging Face Hub")
print(f"Training samples: {len(dataset_from_hub['train'])}")

# %% [markdown]
"""
## Working with your dataset

Once loaded, your dataset behaves like a dictionary of [Hugging Face datasets](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset):
"""

# %%
# Access different splits
print("Available splits:", list(dataset_from_hub.keys()))
print(f"Training samples: {len(dataset_from_hub['train'])}")

# View the first few samples
print("\nFirst 3 training samples:")
train_split = dataset_from_hub["train"][:3]
for i, (utterance, label) in enumerate(zip(train_split["utterance"], train_split["label"], strict=True)):
    print(f"{i+1}. '{utterance}' → label {label}")

# %% [markdown]
"""
### Working with individual samples
"""

# %%
# Access specific samples
first_sample = dataset_from_hub["train"][0]
print(f"First sample: '{first_sample['utterance']}' (label: {first_sample['label']})")

# Slice multiple samples
batch = dataset_from_hub["train"][5:10]
print("\nBatch of 5 samples:")
for utterance, label in zip(batch["utterance"], batch["label"], strict=True):
    print(f"  '{utterance}' → {label}")

# %% [markdown]
"""
## Saving and sharing datasets

### Save to Hugging Face Hub
To share your dataset with others or for reproducibility:
"""

# %%
# dataset.push_to_hub("your-username/your-dataset-name")
# Note: Make sure you're logged in with `huggingface-cli login`

# %% [markdown]
"""
### Save to a local JSON file

You can also save your dataset to a local JSON file for backup or sharing outside the Hugging Face Hub. Use the `to_json` method:
"""

# %%
# Save the dataset to a local JSON file
# dataset_from_hub.to_json("my_banking77_dataset.json")


# %% [markdown]
"""
## Best practices

### 1. **Data quality matters**
- Ensure consistent labeling across your dataset
- Include diverse examples for each intent
- Aim for balanced classes when possible

### 2. **Split your data wisely**
- **Training**: 60-80% of your data (required)
- **Validation**: 10-20% (optional - AutoIntent will create from training if not provided)
- **Test**: 10-20% (highly recommended - keep this frozen for final evaluation)

### 3. **Start small, then scale**
- Begin with a small representative sample (10-20 examples per intent)
- Use AutoIntent to find the best approach
- Scale up with more data once you've validated your setup
- **Tip**: If you have limited data, consider using AutoIntent's augmentation tools (see %mddoclink(rst,augmentation_tutorials.index))
"""

# %% [markdown]
"""
## Next steps

Now that you know how to work with data in AutoIntent, you're ready to explore the different modules and techniques available for intent classification.

**Up next**: Learn how to use individual modules for more control over your intent classification pipeline.

- Next chapter: %mddoclink(notebook,basic_usage.02_modules)
- See also: %mddoclink(notebook,advanced.01_data) covering advanced topics like OOS samples and adding information about intent
"""
