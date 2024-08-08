configs_dir="experiments/testing-metrics/configs"
data_path="data/intent_records/banking77.json"
logs_dir="experiments/testing-metrics/logs"

yaml_files=($(find "$configs_dir" -type f \( -name "*.yaml" -o -name "*.yml" \)))

for filename in "${yaml_files[@]}"; do
    python3 scripts/base_pipeline.py \
        --config-path $filename \
        --data-path $data_path \
        --logs-dir $logs_dir
done