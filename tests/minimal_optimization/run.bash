#!/bin/bash

# Run the first command
poetry run autointent \
    search_space_path="tests/minimal_optimization/configs/multiclass.yaml" \
    multiclass_path="tests/minimal_optimization/data/clinc_subset.json" \
    logs_dir="tests/minimal_optimization/logs" \
    mode="multiclass" \
    device="cpu" \
    log_level="DEBUG" \
    run_name="multiclass-cpu"

# Capture the exit code of the first command
exit_code_1=$?
echo "Exit code of the first command: $exit_code_1"

if [ $exit_code_1 -ne 0 ]; then
    echo "First command failed with exit code $exit_code_1"
    exit $exit_code_1
fi

# Run the second command
poetry run autointent \
    search_space_path=tests/minimal_optimization/configs/multilabel.yaml \
    multiclass_path=tests/minimal_optimization/data/clinc_subset.json \
    logs_dir=tests/minimal_optimization/logs \
    mode=multiclass_as_multilabel \
    device="cpu" \
    log_level="DEBUG" \
    run_name=multilabel-cpu

# Capture the exit code of the second command
exit_code_2=$?
echo "Exit code of the second command: $exit_code_2"

if [ $exit_code_2 -ne 0 ]; then
    echo "Second command failed with exit code $exit_code_2"
    exit $exit_code_2
fi
