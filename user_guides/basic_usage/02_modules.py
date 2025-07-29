# %% [markdown]
"""
# Modules

Modules are the core building blocks of AutoIntent, providing the fundamental functionality for intent classification tasks. This guide will walk you through everything you need to know about using modules effectively in your projects.

## What You'll Learn

By the end of this tutorial, you'll be able to:

- Understand the different types of modules and their roles
- Initialize and configure modules for your specific needs
- Train modules on your data and use them for inference
- Save and load trained modules for reuse
- Debug and inspect module predictions

## Understanding Module Types

AutoIntent provides two complementary types of modules that work together to solve intent classification:

### Scoring Modules
**Purpose**: Convert text utterances into probability distributions over intent classes.

**What they do**: Take raw text as input and output a probability vector where each element represents the likelihood of the utterance belonging to a specific intent class.

**Examples**: %mddoclink(class,modules.scoring,KNNScorer), %mddoclink(class,modules.scoring,LinearScorer), %mddoclink(class,modules.scoring,BertScorer), %mddoclink(class,modules.scoring,CatBoostScorer)

### Decision Modules
**Purpose**: Convert probability scores into final predictions with support for multi-label and out-of-domain detection.

**What they do**: Take probability vectors from scoring modules and apply decision logic to determine the final set of predicted labels.

**Examples**: %mddoclink(class,modules.decision,ArgmaxDecision), %mddoclink(class,modules.decision,ThresholdDecision), %mddoclink(class,modules.decision,TunableDecision)

## Getting Started: Your First Module

Let's start by initializing a simple but effective K-Nearest Neighbors scorer:
"""

# %%
from autointent.modules.scoring import KNNScorer

# Initialize a KNN scorer with basic configuration
scorer = KNNScorer(
    embedder_config="sergeyzh/rubert-tiny-turbo",  # Pre-trained embedding model
    k=5,  # Number of nearest neighbors
)

print(f"Initialized {scorer.__class__.__name__} with k={scorer.k}")

# %% [markdown]
"""
### Embedder Configuration Deep Dive

Most modules in AutoIntent rely on embedders to convert text into numerical representations. We use the powerful **sentence-transformers** library, which provides:

- 🔄 **Automatic device detection** (CUDA, MPS, CPU)
- 🚀 **Optimized inference** with batching and caching
- 🌐 **Access to enumerous models** from Hugging Face Hub
"""

# %% [markdown]
"""
💡 For detailed embedder configuration including custom models, optimization settings, and advanced features, check out our dedicated guide: %mddoclink(notebook,advanced.02_embedder_configuration).
"""

# %% [markdown]
"""
## Loading Training Data

Before we can train our module, we need to prepare the training data. AutoIntent provides convenient data loading utilities:
"""

# %%
from autointent import Dataset

# Load a pre-processed dataset from the hub
dataset = Dataset.from_hub("DeepPavlov/clinc150_subset")

# Let's explore the dataset structure
print("Dataset structure:")
print(f"- Splits: {list(dataset.keys())}")
print(f"- Train split size: {len(dataset['train_0'])}")
print(f"- Sample utterance: '{dataset['train_0']['utterance'][0]}'")
print(f"- Sample label: '{dataset['train_0']['label'][0]}'")

# %% [markdown]
"""
### Understanding the Data Format

The dataset contains text utterances paired with their intent labels. Let's examine a few examples to understand what we're working with:
"""

# %%
# Display sample data
print("Sample training examples:")
for i in range(3):
    utterance = dataset["train_0"]["utterance"][i]
    label = dataset["train_0"]["label"][i]
    print(f'  {i+1}. "{utterance}" → {label}')

print(f"\nTotal unique intents: {len(set(dataset['train_0']['label']))}")

# %% [markdown]
"""
## Training Your Module

Now comes the exciting part - training your module on the data! This is where the module learns to map utterances to intent probabilities:
"""

# %%
import time

print("🚀 Starting training...")
start_time = time.time()

# Train the module on utterances and their corresponding labels
scorer.fit(dataset["train_0"]["utterance"], dataset["train_0"]["label"])

training_time = time.time() - start_time
print(f"✅ Training completed in {training_time:.2f} seconds!")

# %% [markdown]
"""
### What Happens During Training

During the `fit()` process, the module:

1. **Validates setup**: Ensures the data is properly formatted and compatible with this module
2. **Processes text**: Converts utterances into embeddings using the configured embedder
3. **Learns from data**: Builds internal representations (e.g., stores training examples for %mddoclink(class,modules.scoring,KNNScorer), learns weights for %mddoclink(class,modules.scoring,LinearScorer))

The exact process depends on the module type - some modules like %mddoclink(class,modules.scoring,KNNScorer) simply store the training examples, while others like neural networks perform gradient-based optimization.
"""

# %% [markdown]
"""
## Making Predictions

Now that your module is trained, let's see it in action! We can use it to predict intents for new utterances:
"""

# %%
# Test with some example utterances
test_utterances = [
    "hello world!",
    "What's the weather like today?",
    "I want to book a flight to Paris",
    "Play some music please",
]

print("🔮 Making predictions...")
predictions = scorer.predict(test_utterances)

print("\nPrediction results:")
for utterance, prediction in zip(test_utterances, predictions, strict=False):
    print(f'  "{utterance}" → {prediction}')

# %% [markdown]
"""
### Understanding Predictions

The `predict()` method returns the most likely intent class for each input utterance. Behind the scenes, the module:

1. **Embeds the text** using the same embedder used during training
2. **Computes similarities** with training examples (for %mddoclink(class,modules.scoring,KNNScorer)) or applies learned weights (for %mddoclink(class,modules.scoring,LinearScorer))
3. **Returns the highest-scoring intent** as the final prediction

### Batch vs Single Predictions

You can predict on single utterances or batches efficiently:
"""

# %%
# Single prediction
single_prediction = scorer.predict(["How do I reset my password?"])
print(f"Single prediction: {single_prediction[0]}")

# Batch prediction (more efficient for multiple utterances)
batch_predictions = scorer.predict(
    ["Show me my account balance", "What time does the store close?", "Cancel my subscription"]
)
print(f"Batch predictions: {batch_predictions}")

# %% [markdown]
"""
## Saving and Loading Models

One of the most important features for production use is the ability to save trained models and load them later. AutoIntent makes this simple and reliable:

### Saving Your Trained Model
"""

# %%
from pathlib import Path

# Create a directory for saving the model
model_path = Path("my_dumps/knnscorer_clinc150")
model_path.mkdir(parents=True, exist_ok=True)

print(f"💾 Saving model to: {model_path}")
scorer.dump(model_path)
print("✅ Model saved successfully!")

# Let's see what files were created
print("\nSaved files:")
for file in model_path.rglob("*"):
    if file.is_file():
        print(f"  - {file.name}")

# %% [markdown]
"""
### Loading a Saved Model

Loading is just as easy - you can restore the exact same model state without retraining:
"""

# %%
# Load the model from disk
print("📁 Loading saved model...")
loaded_scorer = KNNScorer.load(model_path)
print("✅ Model loaded successfully!")

# Verify it works the same as the original
test_utterance = ["hello world!"]
original_prediction = scorer.predict(test_utterance)
loaded_prediction = loaded_scorer.predict(test_utterance)

print("\nVerification:")
print(f"  Original model: {original_prediction}")
print(f"  Loaded model:   {loaded_prediction}")
print(f"  Identical:      {original_prediction == loaded_prediction}")

# %% [markdown]
"""
## Advanced: Debugging with Rich Output

Many modules provide detailed prediction metadata that's valuable for understanding model behavior and debugging issues:
"""

# %%
# Get detailed prediction information
print("🔍 Analyzing prediction with metadata...")
scores, meta = loaded_scorer.predict_with_metadata(["hello world!"])

print("Detailed prediction analysis:")
print("  Input: 'hello world!'")
print(f"  Prediction: {scores[0]}")

# Display additional metadata if available
print(f"  Similar examples found: {len(meta[0]['neighbors'])}")

# %% [markdown]
"""
### What's in the Metadata?

Different modules provide different types of debugging information:

- **KNN modules** (e.g., %mddoclink(class,modules.scoring,KNNScorer)): Show nearest neighbors from training data
- **Linear modules** (e.g., %mddoclink(class,modules.scoring,LinearScorer)): Display feature importance scores
- **Neural modules** (e.g., %mddoclink(class,modules.scoring,BertScorer)): Provide attention weights and layer activations [planned to implement]

This metadata is crucial for:
- 🐛 **Debugging** unexpected predictions
- 📊 **Model interpretation** and explainability
- 🎯 **Performance optimization** by identifying weak spots
- 🔧 **Feature engineering** based on what the model focuses on

## Summary and Next Steps

Congratulations! You've learned the fundamentals of working with AutoIntent modules:

✅ **What you've accomplished:**
- Understood different module types and their roles
- Configured and initialized modules for your needs
- Trained a module on real data
- Made predictions on new utterances
- Saved and loaded models for reuse
- Explored debugging capabilities with rich output

### 🚀 What's Next?

Now that you understand modules, you're ready for more advanced topics:

1. **Pipeline Automation**: Learn about %mddoclink(notebook,basic_usage.03_automl) to automatically find the best module configurations
2. **Advanced Configuration**: Dive deeper into %mddoclink(notebook,advanced.02_embedder_configuration) for optimal performance
3. **Production Deployment**: Explore inference optimization and serving strategies
4. **Custom Modules**: Build your own modules for specific use cases

### 💡 Key Takeaways

- **Modules are composable**: Combine scoring and decision modules for complex workflows
- **Configuration matters**: Proper embedder and hyperparameter setup significantly impacts performance
- **Debugging is built-in**: Use metadata outputs to understand and improve your models
- **Persistence is seamless**: Save and load models without losing any functionality
"""

# %%
# Clean up the saved files
import shutil

print("🧹 Cleaning up demo files...")
shutil.rmtree(model_path.parent)
print("✅ Cleanup completed!")
