#!/bin/bash

# Check if the required arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <new_metric> <multilabel:true|false>"
  exit 1
fi

NEW_METRIC="$1"
MULTILABEL="$2"

# Determine the correct configuration file based on the multilabel argument
if [ "$MULTILABEL" == "true" ]; then
  CONFIG_PATH="autointent/datafiles/default-multilabel-config.yaml"
elif [ "$MULTILABEL" == "false" ]; then
  CONFIG_PATH="autointent/datafiles/default-multiclass-config.yaml"
else
  echo "Invalid value for <multilabel>. Use 'true' or 'false'."
  exit 1
fi

# Backup the original configuration file
cp "$CONFIG_PATH" "${CONFIG_PATH}.bak"

# Update the metric value where node_type=scoring
yq e "(.nodes[] | select(.node_type == \"scoring\") | .metric) = \"$NEW_METRIC\"" -i "$CONFIG_PATH"

if [ $? -eq 0 ]; then
  echo "Metric value successfully updated to '$NEW_METRIC' in $CONFIG_PATH where node_type=scoring"
else
  echo "Failed to update the metric value in $CONFIG_PATH"
  exit 1
fi
