config="experiments/multilabel/config.yaml"
data_dir="data/multi_label_data"
logs_dir="experiments/multilabel/logs"

datasets_files=($(find "$data_dir" -type f \( -name "*.json" \)))

for filepath in "${datasets_files[@]}"; do
    run_name=$(basename "$filepath")
    run_name="${run_name%.*}"
    python3 scripts/base_pipeline.py \
        --config-path $config \
        --data-path $filepath \
        --logs-dir $logs_dir \
        --run-name $run_name \
        --multilabel
done