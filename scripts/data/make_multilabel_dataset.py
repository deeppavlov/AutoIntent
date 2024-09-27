import json
import os
import random
from argparse import ArgumentParser
from itertools import combinations

import xeger


def sample_unique_tuples(k, n, m):
    all_combinations = list(combinations(range(n), k))
    random.shuffle(all_combinations)
    return all_combinations[:m]


def sample_utterance_from_regexp(intent_record: dict, x: xeger.Xeger):
    n_templates = len(intent_record["regexp_full_match"])
    i_template = random.randint(0, n_templates - 1)
    res = x.xeger(intent_record["regexp_full_match"][i_template])
    return res.strip()


def sample_multilabel_utterances(intent_records: list[dict], n_samples, n_labels, seed=0):
    random.seed(seed)
    x = xeger.Xeger()
    x.seed(seed)
    n_given_intents = len(intent_records)
    res = []
    for t in sample_unique_tuples(n_labels, n_given_intents, n_samples):
        sampled_utterances = [sample_utterance_from_regexp(intent_records[i], x) for i in t]
        utterance = ". ".join(sampled_utterances)
        res.append({"utterance": utterance, "labels": t})
    return res


def get_multilabel_version(intent_records, config_string, seed):
    config = json.loads(config_string)
    res = []
    for i in range(len(config)):
        res.extend(sample_multilabel_utterances(intent_records, n_samples=int(config[i]), n_labels=i + 1, seed=seed))
    return res


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="path to intent records")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help='config string like "[20, 40, 20, 10]" means 20 one-label examples, 40 two-label examples, 20 three-label examples, 10 four-label examples',
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    intent_records = json.load(open(args.input_path))

    res = get_multilabel_version(intent_records, args.config, args.seed)

    dirname = os.path.dirname(args.output_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    json.dump(res, open(args.output_path, "w"), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
