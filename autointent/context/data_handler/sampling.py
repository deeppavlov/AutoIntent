import random

import sre_yield


def distribute_shots(n: int, k: int) -> list[int]:
    """randomly distribute `k` samples among `n` bins"""
    samples_per_bin = [0] * n
    for _ in range(k):
        i_bin = random.randint(0, n - 1)
        samples_per_bin[i_bin] += 1
    return samples_per_bin


def generate_from_template(template: str, n: int) -> list[str]:
    """generate `n` samples from `template`, or fewer if it's impossible"""
    return list(sre_yield.AllStrings(template))[:n]


def generate_from_templates(patterns: list[str], n_shots: int) -> list[str]:
    shots_per_pattern = distribute_shots(len(patterns), n_shots)
    res = []
    for pattern, n in zip(patterns, shots_per_pattern, strict=False):
        new_samples = generate_from_template(pattern, n)
        res.extend(new_samples)
    return res


def sample_from_regex(intent_records: list[dict], n_shots: int) -> list[dict]:
    for intent in intent_records:
        new_samples = generate_from_templates(intent["regexp_full_match"], n_shots)
        intent["sample_utterances"].extend(new_samples)
    return intent_records
