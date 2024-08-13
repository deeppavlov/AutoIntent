import random

from xeger import Xeger


def distribute_shots(n, k):
    """randomly distribute `k` samples among `n` bins"""
    samples_per_bin = [0] * n
    for _ in range(k):
        i_bin = random.randint(0, n - 1)
        samples_per_bin[i_bin] += 1
    return samples_per_bin


def sample_xeger(x: Xeger, patterns: list[str], shots_per_pattern: list[int]):
    res = []
    for pattern, n_shots in zip(patterns, shots_per_pattern):
        new_samples = [x.xeger(pattern) for _ in range(n_shots)]
        res.extend(new_samples)
    return res


if __name__ == "__main__":
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, default="data/intent_records/dream.json")
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    intent_records = json.load(open(args.data_path))
    x = Xeger()

    for intent in intent_records:
        n_patterns = len(intent["regexp_full_match"])
        shots_per_pattern = distribute_shots(n_patterns, args.n_shots)
        new_samples = sample_xeger(x, intent["regexp_full_match"], shots_per_pattern)
        intent["sample_utterances"].extend(new_samples)

    json.dump(intent_records, open(args.output_path, "w"), indent=4, ensure_ascii=False)
