# %% [markdown]
"""
# Command Line Interface for Pipeline Auto Configuration

## Data

Just like with Python API, you can run an automatic pipeline configuration with just a prepared data set.

You can use local JSON file:
```bash
autointent data.train_path="path/to/my.json"
```

Or dataset from Hugging Face hub:
```bash
autointent data.train_path="AutoIntent/banking77"
```

## Search Space

You can provide custom search space, saved as YAML file (as explained in %mddoclink(notebook,cli.01_search_space)):
```bash
autointent data.train_path="AutoIntent/banking77" task.search_space_path="path/to/my/search/space.yaml"
```

## Logging Level

AutoIntent provides comprehensive logs. You can enable it by changing default logging level:
```bash
autointent data.train_path="AutoIntent/banking77" hydra.job_logging.root.level=INFO
```

## All Options

```yaml
data:
# Path to a json file with training data. Set to "default" to use AutoIntent/clinc150_subset from HF hub.
  train_path: ???

# Path to a json file with test records. Skip this option if you want to use a random subset of the
# training sample as test data.
  test_path: null

# Set to true if your data is multiclass but you want to train the multilabel classifier.
  force_multilabel: false

task:
# Path to a yaml configuration file that defines the optimization search space.
# Omit this to use the default configuration.
  search_space_path: null
logs:
# Name of the run prepended to optimization assets dirname (generated randomly if omitted)
  run_name: "awful_hippo_10-30-2024_19-42-12"

# Location where to save optimization logs that will be saved as `<logs_dir>/<run_name>_<cur_datetime>/logs.json`.
# Omit to use current working directory. <-- on Windows it is not correct
  dirpath: "/home/user/AutoIntent/awful_hippo_10-30-2024_19-42-12"

  dump_dir: "/home/user/AutoIntent/runs/awful_hippo_10-30-2024_19-42-12/modules_dumps"

vector_index:
# Location where to save faiss database file. Omit to use your system's default cache directory.
  db_dir: null

# Specify device in torch notation
  device: cpu

augmentation:
# Number of shots per intent to sample from regular expressions. This option extends sample utterance
# within multiclass intent records.
  regex_sampling: 0

# Config string like "[20, 40, 20, 10]" means 20 one-label examples, 40 two-label examples, 20 three-label examples,
# 10 four-label examples. This option extends multilabel utterance records.
  multilabel_generation_config: null

embedder:
# batch size for embedding computation.
  batch_size: 1
# sentence length limit for embedding computation
  max_length: null

#Affects the randomness
seed: 0

# String from {DEBUG,INFO,WARNING,ERROR,CRITICAL}. Omit to use ERROR by default.
hydra.job_logging.root.level: "ERROR"
```

## Run from Config File

Create a yaml file in a separate folder with the following structure **my_config.yaml**:
```yaml
defaults:
- optimization_config
- _self_
- override hydra/job_logging: custom

# put the configuration options you want to override here. The full structure is presented above.
# Here is just an example with the same options as for the command line variant above.
embedder:
embedder_batch_size: 32
```

Launch AutoIntent:
```bash
autointent --config-path=/path/to/config/directory --config-name=my_config
```

Important:
* specify the full path in the config-path option.
* do not use tab in the yaml file.
* it is desirable that the file name differs from
optimization_config.yaml to avoid warnings from hydra

You can use a combination of Option 1 and 2. Command line options have the highest priority.

Example configs are stored in our GitHub repository in [example_configs](https://github.com/deeppavlov/AutoIntent/tree/dev/example_configs).
"""
