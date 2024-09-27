import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from random import seed, shuffle


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--k-shots", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed(args.seed)

    utterance_records = json.load(open(args.input_path))
    shuffle(utterance_records)

    shots_per_class_gathered = defaultdict(int)
    shots_ids = []
    for i, record in enumerate(utterance_records):
        if is_shot_useful(record, shots_per_class_gathered, args.k_shots):
            update_counter(shots_per_class_gathered, record["labels"])
            shots_ids.append(i)

    res = [utterance_records[i] for i in shots_ids]
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    json.dump(res, open(args.output_path, "w"), indent=4, ensure_ascii=False)



def update_counter(counter: defaultdict, labels: list[int]):
    for lab in labels:
        counter[lab] += 1


def is_shot_useful(record: dict, counter: defaultdict, k_shots: int):
    if record["utterance"] == "":
        return False
    return any(counter[lab] < k_shots for lab in record["labels"])


if __name__ == "__main__":
    main()
