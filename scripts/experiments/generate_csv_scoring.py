import os
import json
import pandas as pd
import argparse

def process_logs(base_dir, filter_keyword, output_csv):
    data = []

    for root, dirs, files in os.walk(base_dir):
        if filter_keyword in root and 'logs.json' in files:
            dataset_name = os.path.basename(root)
            dataset_name = dataset_name.split("_scoring")[0]
            log_path = os.path.join(root, 'logs.json')

            with open(log_path, 'r') as f:
                log_data = json.load(f)

            for metric_type, metrics in log_data['metrics'].items():
                for config in log_data['configs'][metric_type]:
                    method = config['module_type']
                    metric_name = config['metric_name']
                    metric_value = config['metric_value']

                    row = {
                        'dataset': dataset_name,
                        'method': method,
                        'metric': metric_name,
                        'metric_value': metric_value
                    }

                    for param, value in config['module_params'].items():
                        row[param] = value

                    data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV file '{output_csv}' successfully created!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process logs and generate a CSV file.")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory where folders are located")
    parser.add_argument('--filter_keyword', type=str, required=True, help="Keyword to filter folders (e.g., 'multilabel')")
    parser.add_argument('--output_csv', type=str, required=True, help="Output CSV file name")
    args = parser.parse_args()

    process_logs(args.base_dir, args.filter_keyword, args.output_csv)
