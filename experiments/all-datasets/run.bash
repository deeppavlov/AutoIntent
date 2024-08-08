config="experiments/all-datasets/config.yaml"
data_dir="data/intent_records"
logs_dir="experiments/all-datasets/logs"

datasets_files=($(find "$data_dir" -type f \( -name "*.json" \)))

for filepath in "${datasets_files[@]}"; do
    run_name=$(basename "$filepath")
    run_name="${run_name%.*}"
    python3 scripts/base_pipeline.py \
        --config-path $config \
        --data-path $filepath \
        --logs-dir $logs_dir \
        --run-name $run_name
done