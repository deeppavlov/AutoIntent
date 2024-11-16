#!/bin/bash

# Check for the required arguments
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <DATA_PATH> <LOG_PATH> <USE_MULTILABEL>"
  exit 1
fi

# Read arguments
DATA_PATH="$1"
LOG_PATH="$2"
USE_MULTILABEL="$3"
CONFIG_SCRIPT_PATH="scripts/experiments/update_metric.sh"

# Define metrics for multilabel and multiclass
if [ "$USE_MULTILABEL" = true ]; then
  METRICS=(
    "scoring_accuracy"
    "scoring_f1"
    "scoring_log_likelihood"
    "scoring_precision"
    "scoring_recall"
    "scoring_roc_auc"
    "scoring_neg_ranking_loss"
    "scoring_neg_coverage"
    "scoring_hit_rate"
  )
else
  METRICS=(
    "scoring_accuracy"
    "scoring_f1"
    "scoring_log_likelihood"
    "scoring_precision"
    "scoring_recall"
    "scoring_roc_auc"
  )
fi

# Iterate through each metric
for METRIC in "${METRICS[@]}"; do
  echo "Processing with metric: $METRIC"

  for FILE in "$DATA_PATH"/*.json; do
    FILENAME=$(basename "$FILE" .json)
    DATASET_NAME=$(echo "$FILENAME" | sed 's/_fix.*//')

    # Update the metric in the configuration file
    echo "Updating metric for dataset: $DATASET_NAME"
    $CONFIG_SCRIPT_PATH "$METRIC" "$USE_MULTILABEL"
    if [ $? -ne 0 ]; then
      echo "Error updating metric for $DATASET_NAME with metric: $METRIC. Exiting."
      exit 1
    fi

    rm -rf runs/
    rm -rf outputs/
    rm -rf vector_db_*

    echo "Processing dataset: $DATASET_NAME with metric: $METRIC"
    autointent data.train_path="$FILE" \
               logs.dirpath="$LOG_PATH/${DATASET_NAME}_${METRIC}" \
               seed=42 \
               vector_index.device=cuda \
               hydra.job_logging.root.level=INFO \
               data.force_multilabel="$USE_MULTILABEL"

    if [ $? -ne 0 ]; then
      echo "Error encountered while processing $FILE with metric: $METRIC. Exiting."
      exit 1
    else
      echo "Successfully processed $FILE with metric: $METRIC"
    fi
  done
done

echo "All datasets processed successfully for all metrics."
