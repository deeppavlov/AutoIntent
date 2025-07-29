# %% [markdown]
"""
# Data

This chapter covers advanced data handling techniques in AutoIntent that go beyond basic dataset creation. You'll learn how to handle out-of-scope samples, enrich your data with intent metadata, and leverage advanced features for robust intent classification systems.
"""

# %% [markdown]
"""
**Prerequisites**: Complete the %mddoclink(notebook,basic_usage.01_data) tutorial first.
"""

# %%
import datasets

from autointent import Dataset

# %%
datasets.logging.disable_progress_bar()  # disable tqdm outputs for cleaner output

# %% [markdown]
"""
## Handling Out-of-Scope (OOS) Samples

Out-of-scope detection is crucial for robust intent classification systems. Users often say things that don't match any of your predefined intents, and your system needs to handle these gracefully.

### What are Out-of-Scope Samples?

Out-of-scope (OOS) samples are utterances that don't belong to any of your defined intent classes. For example, in a banking chatbot, "What's the weather like?" would be out-of-scope.
"""

# %%
# Create a dataset with out-of-scope samples
banking_with_oos = {
    "train": [
        # In-domain samples
        {"utterance": "What's my account balance?", "label": 0},
        {"utterance": "Check my current balance", "label": 0},
        {"utterance": "I want to transfer money to my friend", "label": 1},
        {"utterance": "How do I send funds to someone?", "label": 1},
        {"utterance": "Cancel my last transaction", "label": 2},
        {"utterance": "Reverse the payment I just made", "label": 2},
        # Out-of-scope samples (no label field)
        {"utterance": "What's the weather like today?"},
        {"utterance": "Tell me a joke"},
        {"utterance": "How do I cook pasta?"},
        {"utterance": "What time is it?"},
        {"utterance": "I love pizza"},
    ],
    "test": [
        {"utterance": "Show me my current balance", "label": 0},
        {"utterance": "Transfer $100 to my savings", "label": 1},
        {"utterance": "Stop my recent payment", "label": 2},
        {"utterance": "What's your favorite movie?"},  # OOS
        {"utterance": "How's the traffic today?"},  # OOS
    ],
    "intents": [
        {"id": 0, "name": "balance_inquiry"},
        {"id": 1, "name": "money_transfer"},
        {"id": 2, "name": "transaction_cancellation"},
    ],
}

dataset_with_oos = Dataset.from_dict(banking_with_oos)
print("✅ Dataset with OOS samples created")
print(f"Available splits: {list(dataset_with_oos.keys())}")

# %% [markdown]
"""
### Advanced OOS Strategies

For robust systems, you'll want to carefully curate your OOS samples:

1. **Domain-adjacent samples**: Include utterances that are close to your domain but still out-of-scope
2. **Common conversational patterns**: Add greetings, small talk, and common user behaviors
3. **Edge cases**: Include borderline cases that might confuse your model
"""

# %%
# Example of well-curated OOS samples for a banking domain
sophisticated_oos_data = {
    "train": [
        # In-scope samples
        {"utterance": "What's my account balance?", "label": 0},
        {"utterance": "I want to transfer money", "label": 1},
        # Sophisticated out-of-scope samples
        {"utterance": "Hello, how are you?"},  # Greeting
        {"utterance": "Thanks for your help!"},  # Courtesy
        {"utterance": "What other services do you offer?"},  # Domain-adjacent
        {"utterance": "I'm having trouble with the app"},  # Technical support (different domain)
        {"utterance": "Can you recommend a good investment?"},  # Financial advice (borderline)
        {"utterance": "What are your business hours?"},  # Information request (different domain)
    ],
    "intents": [
        {"id": 0, "name": "balance_inquiry"},
        {"id": 1, "name": "money_transfer"},
    ],
}

sophisticated_dataset = Dataset.from_dict(sophisticated_oos_data)

# %% [markdown]
"""
## Enriching Data with Intent Metadata

Intent metadata allows you to provide additional information about your intents that can be leveraged by various AutoIntent modules for improved performance.
"""

# %% [markdown]
"""
### Intent Metadata Example

Here's an example showing how to add metadata to your intents:
"""

# %%
# Create a dataset with rich intent metadata
comprehensive_banking_data = {
    "train": [
        {"utterance": "What's my account balance?", "label": 0},
        {"utterance": "Check my current balance", "label": 0},
        {"utterance": "How much money do I have?", "label": 0},
        {"utterance": "I want to transfer money", "label": 1},
        {"utterance": "Send funds to my friend", "label": 1},
        {"utterance": "Make a payment to someone", "label": 1},
        {"utterance": "Cancel my last transaction", "label": 2},
        {"utterance": "Reverse this payment", "label": 2},
        {"utterance": "Stop my transfer", "label": 2},
        {"utterance": "I need help with my account", "label": 3},
        {"utterance": "Can someone assist me?", "label": 3},
        {"utterance": "I have a question about my account", "label": 3},
    ],
    "intents": [
        {
            "id": 0,
            "name": "balance_inquiry",
            "description": "User wants to check their account balance or available funds",
        },
        {
            "id": 1,
            "name": "money_transfer",
            "description": "User wants to transfer money or make a payment to another person or account",
        },
        {
            "id": 2,
            "name": "transaction_cancellation",
            "description": "User wants to cancel, reverse, or stop a transaction or payment",
        },
        {
            "id": 3,
            "name": "general_help",
            "description": "User is requesting general assistance or has a question",
        },
    ],
}

rich_dataset = Dataset.from_dict(comprehensive_banking_data)
print("✅ Dataset with rich intent metadata created")

# %% [markdown]
"""
### Understanding Intent Metadata Fields

Let's examine what each metadata field does and how AutoIntent modules use them:
"""

# %%
# Examine the intent metadata
print("Intent metadata breakdown:\n")
for intent in rich_dataset.intents:
    print(f"🎯 Intent: {intent.name} (ID: {intent.id})")
    print(f"   Description: {intent.description}")
    print()

# %% [markdown]
"""
### How Modules Use Intent Metadata

- **`name`**: Human-readable intent names for interpretability and debugging, also can be used by AutoIntent augmentation methods
- **`description`**: Used by %mddoclink(class,modules.scoring,BiEncoderDescriptionScorer), %mddoclink(class,modules.scoring,CrossEncoderDescriptionScorer), %mddoclink(class,modules.scoring,LLMDescriptionScorer) to calculate semantic similarity between utterances and intent descriptions

**Pro tip**: Well-crafted descriptions significantly improve performance for description-based scoring modules, especially with limited training data.
"""

# %% [markdown]
"""
## Advanced Dataset Manipulation

### Working with Large Datasets

For systems with large datasets, you'll want efficient ways to manipulate and analyze your data:
"""

# %%
# Load a larger dataset for demonstration
dataset = Dataset.from_hub("DeepPavlov/clinc150_subset")

# Dataset analysis
print("📊 Dataset Analysis")
print(f"Dataset splits: {list(dataset.keys())}")
print(f"Total training samples: {len(dataset['train_0']) + len(dataset['train_1'])}")
print(f"Number of intents: {len(dataset.intents)}")

# Examine class distribution
from collections import Counter

label_counts = Counter(dataset["train_0"]["label"])
print("\nClass distribution (top 5):")
for label, count in label_counts.most_common(5):
    intent_name = dataset.intents[label].name
    print(f"  {intent_name} (label {label}): {count} samples")

# %% [markdown]
"""
### Custom Data Processing

You can process your datasets using the underlying Hugging Face datasets functionality:
"""

# %%
# Example: Filter samples by length
short_utterances = dataset["train_0"].filter(lambda x: len(x["utterance"].split()) <= 5)
print(f"Short utterances (≤5 words): {len(short_utterances)} samples")


# Example: Add computed features
def add_utterance_length(example):
    example["utterance_length"] = len(example["utterance"].split())
    return example


enriched_train = dataset["train_0"].map(add_utterance_length)
print(f"Added utterance_length feature to {len(enriched_train)} samples")

# Show example with new feature
sample = enriched_train[0]
print(f"Sample: '{sample['utterance']}' (length: {sample['utterance_length']} words)")

# %% [markdown]
"""
### Creating Custom Splits

For advanced experimentation, you might want to create custom data splits:
"""


# %%
# Example: Create a custom split based on utterance characteristics
def create_length_based_splits(dataset_split, short_threshold=5, long_threshold=10):
    """Split data based on utterance length for targeted evaluation."""

    def is_short(example):
        return len(example["utterance"].split()) <= short_threshold

    def is_long(example):
        return len(example["utterance"].split()) >= long_threshold

    short_split = dataset_split.filter(is_short)
    long_split = dataset_split.filter(is_long)

    return short_split, long_split


short_test, long_test = create_length_based_splits(dataset["test"])
print("Custom splits created:")
print(f"  Short utterances: {len(short_test)} samples")
print(f"  Long utterances: {len(long_test)} samples")

# %% [markdown]
"""
## Next Steps

You now understand advanced data handling in AutoIntent, including:

- ✅ Out-of-scope sample handling for robust intent classification
- ✅ Intent metadata for improved model performance
- ✅ Advanced dataset manipulation and analysis techniques

**What's next:**
- Explore %mddoclink(notebook,advanced.03_automl) for advanced AutoML techniques
- Learn about %mddoclink(rst,augmentation_tutorials.index) to expand your datasets
- See %mddoclink(notebook,advanced.04_reporting) for comprehensive model evaluation

**Pro tip**: Start with a small, well-curated dataset with good intent descriptions, then scale up using AutoIntent's optimization capabilities to find the best approach for your specific use case.
"""
