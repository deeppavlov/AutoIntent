import random

import sre_yield


def distribute_shots(n, k):
    """randomly distribute `k` samples among `n` bins"""
    samples_per_bin = [0] * n
    for _ in range(k):
        i_bin = random.randint(0, n - 1)
        samples_per_bin[i_bin] += 1
    return samples_per_bin


def generate_from_template(template, n):
    """generate `n` samples from `template`, or fewer if its impossible"""
    iterator = iter(sre_yield.AllStrings(template))
    res = []
    for _ in range(n):
        try:
            res.append(next(iterator))
        except StopIteration:
            break
    return res


def generate_from_templates(patterns: list[str], shots_per_pattern: list[int]):
    res = []
    for pattern, n_shots in zip(patterns, shots_per_pattern):
        new_samples = generate_from_template(pattern, n_shots)
        res.extend(new_samples)
    return res


def sample_from_regex(intent_records: list[dict], n_shots, seed=0):
    random.seed(seed)
    for intent in intent_records:
        n_patterns = len(intent["regexp_full_match"])
        shots_per_pattern = distribute_shots(n_patterns, n_shots)
        new_samples = generate_from_templates(intent["regexp_full_match"], shots_per_pattern)
        intent["sample_utterances"].extend(new_samples)
