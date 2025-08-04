# %% [markdown]
"""
# AutoML Pipeline Configuration

AutoML (Automated Machine Learning) in AutoIntent allows you to automatically find the best configuration for your intent classification pipeline. Instead of manually tuning hyperparameters and selecting components, AutoML explores different combinations to find the optimal setup for your specific dataset.
"""

# %%
from autointent import Pipeline

# %% [markdown]
"""
In this tutorial, we'll walk through the pipeline auto-configuration process step by step. We'll learn how to:

- Use predefined search spaces and presets
- Customize search configurations
- Set up logging and validation strategies
- Run the optimization process
- Save and load optimized pipelines

Let's start by loading a small subset of the popular `clinc150` dataset for demonstration.
"""

# %%
from autointent import Dataset

# Load the dataset from Hugging Face hub
dataset = Dataset.from_hub("DeepPavlov/clinc150_subset")
print(f"Dataset contains {len(dataset)} splits")
dataset

# %% [markdown]
"""
Let's examine the structure of our dataset by looking at a sample utterance:
"""

# %%
sample = dataset["train_0"][0]
print(f"Sample utterance: '{sample['utterance']}'")
print(f"Intent label: '{sample['label']}'")
sample

# %% [markdown]
"""
## Search Space

AutoIntent provides default search spaces. One can utilize them by constructing %mddoclink(class,,Pipeline) with factory %mddoclink(method,Pipeline,from_preset):
"""

# %%
pipeline = Pipeline.from_preset("classic-light")

# %% [markdown]
"""
You can inspect the structure and default values of any preset:
"""

# %%
from pprint import pprint

from autointent.utils import load_preset

preset = load_preset("classic-light")
pprint(preset)

# %% [markdown]
"""
### Customizing Search Spaces

The search space can be customized to fit your specific needs. For example, you can modify hyperparameter ranges:
"""

# %%
# Example: modify the maximum k value for KNN-based components
preset["search_space"][0]["search_space"][0]["k"]["high"] = 10
custom_pipeline = Pipeline.from_optimization_config(preset)

# %% [markdown]
"""
See tutorial %mddoclink(notebook,advanced.03_search_space_configuration) on how the search space is structured.
"""

# %% [markdown]
"""
## Logging and Storage Configuration

During the AutoML process, you'll want to control what artifacts are saved and where they're stored. The %mddoclink(class,configs,LoggingConfig) allows you to specify:

- `project_dir`: Directory where results will be saved
- `dump_modules`: Whether to save trained model files
- `clear_ram`: Whether to clear models from memory after training to save RAM
"""

# %%
from pathlib import Path

from autointent.configs import LoggingConfig

logging_config = LoggingConfig(
    project_dir=Path.cwd() / "runs",  # Save results to 'runs' directory
    dump_modules=False,  # Don't save large model files
    clear_ram=False,  # Keep models in memory for inference
)
custom_pipeline.set_config(logging_config)

# %% [markdown]
"""
## Model Configuration

You can specify which transformer models to use for text embeddings and cross-encoding. This is useful when you want to:

- Use smaller/faster models for experimentation
- Apply domain-specific pre-trained models
- Control model parameters like tokenizer settings
"""

# %%
from autointent.configs import CrossEncoderConfig, EmbedderConfig, TokenizerConfig

# Configure embedding model (used for vector representations)
custom_pipeline.set_config(EmbedderConfig(model_name="prajjwal1/bert-tiny"))

# Configure cross-encoder model (used for scoring text pairs)
custom_pipeline.set_config(
    CrossEncoderConfig(model_name="cross-encoder/ms-marco-MiniLM-L2-v2", tokenizer_config=TokenizerConfig(max_length=8))
)

# %% [markdown]
"""
See the documentation for %mddoclink(class,configs,EmbedderConfig) and %mddoclink(class,configs,CrossEncoderConfig) for all available customization options.
"""

# %% [markdown]
"""
## Validation Strategy

Choose between two validation approaches based on your dataset size:

**Hold-out validation** (default): Uses separate train/validation splits. Best when you have plenty of data.

**Cross-validation**: Splits data into k folds for more robust evaluation. Better for smaller datasets as it uses all data for both training and validation.
"""

# %%
from autointent.configs import DataConfig

# Use 3-fold cross-validation for better performance on small datasets
custom_pipeline.set_config(DataConfig(scheme="cv", n_folds=3))

# %% [markdown]
"""
See the docs for %mddoclink(class,configs,DataConfig) for other options available to customize.
"""

# %% [markdown]
"""
## Complete Example

Let's put everything together in a comprehensive example that demonstrates the full AutoML workflow:
"""

# %%
from autointent import Dataset, Pipeline
from autointent.configs import LoggingConfig
from autointent.utils import load_preset

# Step 1: Load your dataset
dataset = Dataset.from_hub("DeepPavlov/clinc150_subset")
print(f"Loaded dataset with {len(dataset)} splits")

# Step 2: Load and customize a preset configuration
preset = load_preset("classic-light")
# You can modify the preset here if needed
# preset["search_space"][0]["search_space"][0]["k"]["high"] = 5

# Step 3: Create pipeline from the configuration
pipeline = Pipeline.from_optimization_config(preset)

# Step 4: Configure logging and storage
logging_config = LoggingConfig(
    dump_modules=True,  # Save trained models for later use
    clear_ram=False,  # Keep models in memory for immediate inference
)
pipeline.set_config(logging_config)

# Step 5: Run AutoML optimization
print("Starting AutoML optimization...")
context = pipeline.fit(dataset)
print("✅ AutoML optimization completed!")

# Step 6: Test the optimized pipeline
test_utterances = ["hello world!", "I want to transfer money", "book a flight"]
predictions = pipeline.predict(test_utterances)
print(f"Predictions: {predictions}")

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
pipeline.dump()

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
