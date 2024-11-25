from autointent.context.data_handler._sampling import sample_from_regex

if __name__ == "__main__":
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, default="data/intent_records/dream.json")
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    intent_records = json.load(open(args.input_path))

    sample_from_regex(intent_records, args.n_shots)

    json.dump(intent_records, open(args.output_path, "w"), indent=4, ensure_ascii=False)
