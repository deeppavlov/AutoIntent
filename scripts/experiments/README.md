## Overview
This script automates the process of evaluating datasets with multiple metrics, either in a multilabel or multiclass configuration. It iterates through datasets in a specified directory and applies a set of metrics to each dataset. The script is designed to work with `autointent` and updates configuration files before processing each dataset.

---

## Features
- Processes datasets for **multilabel** or **multiclass** scenarios based on user input.
- Supports multiple metrics:
  - **Multilabel metrics**:
    - `scoring_accuracy`
    - `scoring_f1`
    - `scoring_log_likelihood`
    - `scoring_precision`
    - `scoring_recall`
    - `scoring_roc_auc`
    - `scoring_neg_ranking_loss`
    - `scoring_neg_coverage`
    - `scoring_hit_rate`
  - **Multiclass metrics**:
    - `scoring_accuracy`
    - `scoring_f1`
    - `scoring_log_likelihood`
    - `scoring_precision`
    - `scoring_recall`
    - `scoring_roc_auc`
- Automatically handles configuration updates using `update_metric.sh`.
- Logs processing results and skips datasets gracefully on errors.

---

## Requirements
- **Dependencies**:
  - `autointent` must be installed and available in the PATH.
  - `yq` is required for processing YAML files. Ensure it is installed and available in the PATH.

    ### Installing `yq`

    #### Linux
    ```
    wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq
    chmod +x /usr/bin/yq
    ```
    #### macOS
    ```
    brew install yq
    ```

- **Input Files**:
  - JSON files located in the directory specified by `<DATA_PATH>`.

---

## Usage
From root repo:
```
sh scripts/experiments/generate_experiments.sh <DATA_PATH> <LOG_PATH> <USE_MULTILABEL>
```
Parameters

    <DATA_PATH>: Path to the directory containing dataset JSON files.
    <LOG_PATH>: Directory where logs for each dataset will be saved.
    <USE_MULTILABEL>: Boolean flag (true or false) indicating whether to use multilabel metrics.

## Example
```
sh scripts/experiments/generate_experiments.sh data/intent_records_regexp/ experiments/dnnc/ false
```

This command processes all JSON files in `data/intent_records_regexp/` using multiclass metrics, saving logs in `experiments/dnnc/`.

## Notes

- Ensure the path to update_metric.sh is correct. Adjust the CONFIG_SCRIPT_PATH variable if needed.
