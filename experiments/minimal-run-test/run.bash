poetry run autointent \
    --config-path experiments/minimal-run-test/configs/multiclass.yaml \
    --multiclass-path experiments/minimal-run-test/data/clinc_subset.json \
    --logs-dir experiments/minimal-run-test/logs \
    --mode multiclass \
    --device "cpu" \
    --verbose \
    --run-name multiclass-cpu

poetry run autointent \
    --config-path experiments/minimal-run-test/configs/multilabel.yaml \
    --multiclass-path experiments/minimal-run-test/data/clinc_subset.json \
    --logs-dir experiments/minimal-run-test/logs \
    --mode multiclass_as_multilabel \
    --device "cpu" \
    --verbose \
    --run-name multilabel-cpu
