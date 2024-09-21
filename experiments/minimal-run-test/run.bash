#!/bin/bash

# Run the first command
poetry run autointent \
    --config-path experiments/minimal-run-test/configs/multiclass.yaml \
    --multiclass-path experiments/minimal-run-test/data/clinc_subset.json \
    --logs-dir experiments/minimal-run-test/logs \
    --mode multiclass \
    --device "cpu" \
    --verbose \
    --run-name multiclass-cpu

# Capture the exit code of the first command
exit_code_1=$?
echo "Exit code of the first command: $exit_code_1"

if [ $exit_code_1 -ne 0 ]; then
    echo "First command failed with exit code $exit_code_1"
    exit $exit_code_1
fi

# Run the second command
poetry run autointent \
    --config-path experiments/minimal-run-test/configs/multilabel.yaml \
    --multiclass-path experiments/minimal-run-test/data/clinc_subset.json \
    --logs-dir experiments/minimal-run-test/logs \
    --mode multiclass_as_multilabel \
    --device "cpu" \
    --verbose \
    --run-name multilabel-cpu

# Capture the exit code of the second command
exit_code_2=$?
echo "Exit code of the second command: $exit_code_2"

if [ $exit_code_2 -ne 0 ]; then
    echo "Second command failed with exit code $exit_code_2"
    exit $exit_code_2
fi
