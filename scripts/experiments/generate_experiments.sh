#!/bin/bash

DATA_PATH="experiments/intent_description"
LOG_PATH="experiments/intent_description/multilabel"
METRIC="scoring_hit_rate"
USE_MULTILABEL=true
CONFIG_SCRIPT_PATH="./update_metric.sh"

for FILE in "$DATA_PATH"/*.json; do
  FILENAME=$(basename "$FILE" .json)
  DATASET_NAME=$(echo "$FILENAME" | sed 's/_fix.*//')

  # Determine the appropriate multilabel flag for the metric update script
  if [ "$USE_MULTILABEL" = true ]; then
    MULTILABEL_ARG="true"
  else
    MULTILABEL_ARG="false"
  fi

  # Update the metric in the configuration file
  echo "Updating metric for dataset: $DATASET_NAME"
  $CONFIG_SCRIPT_PATH "$METRIC" "$MULTILABEL_ARG"
  if [ $? -ne 0 ]; then
    echo "Error updating metric for $DATASET_NAME. Exiting."
    exit 1
  fi

  rm -rf runs/

  echo "Processing dataset: $DATASET_NAME"
  autointent data.train_path="$FILE" \
             logs.dirpath="$LOG_PATH/${DATASET_NAME}_${METRIC}" \
             seed=42 \
             vector_index.device=cuda \
             hydra.job_logging.root.level=INFO \
             data.force_multilabel="$USE_MULTILABEL"

  if [ $? -ne 0 ]; then
    echo "Error encountered while processing $FILE. Exiting."
    exit 1
  else
    echo "Successfully processed $FILE"
  fi
done

echo "All datasets processed successfully."
