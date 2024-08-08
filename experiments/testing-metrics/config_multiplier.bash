#!/bin/bash

# Define the metric lists
retrieval_metrics=(
    "retrieval_map"
    "retrieval_hit_rate"
    "retrieval_precision"
    "retrieval_ndcg"
    "retrieval_mrr"
)

scoring_metrics=(
    "scoring_neg_cross_entropy"
    "scoring_roc_auc"
    "scoring_accuracy"
    "scoring_f1"
    "scoring_precision"
    "scoring_recall"
)

prediction_metrics=(
    "prediction_accuracy"
    "prediction_roc_auc"
    "prediction_precision"
    "prediction_recall"
    "prediction_f1"
)

# Input: Path to the original YAML config
original_yaml="$1"
output_dir="$2"
mkdir -p $output_dir

# Function to update the YAML file with new metrics
update_yaml() {
    local retrieval_metric="$1"
    local scoring_metric="$2"
    local prediction_metric="$3"
    local output_file="$4"

    # Read the original YAML content
    yaml_content=$(<"$original_yaml")

    # Update the metrics in the YAML content
    yaml_content=$(echo "$yaml_content" | sed "s/metric: retrieval_metric/metric: $retrieval_metric/")
    yaml_content=$(echo "$yaml_content" | sed "s/metric: scoring_metric/metric: $scoring_metric/")
    yaml_content=$(echo "$yaml_content" | sed "s/metric: prediction_metric/metric: $prediction_metric/")

    # Write the updated content to the output file
    echo "$yaml_content" > "$output_file"
}

# Generate all combinations and create copies of the YAML file
for retrieval_metric in "${retrieval_metrics[@]}"; do
    for scoring_metric in "${scoring_metrics[@]}"; do
        for prediction_metric in "${prediction_metrics[@]}"; do
            # Define the output file name
            output_file="${output_dir}/config_${retrieval_metric}_${scoring_metric}_${prediction_metric}.yaml"

            # Update the YAML file with the new metrics
            update_yaml "$retrieval_metric" "$scoring_metric" "$prediction_metric" "$output_file"

            echo "Created $output_file"
        done
    done
done
