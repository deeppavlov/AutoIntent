"""One-shot extractor: pulls DeepPavlov/clinc150 and writes a stable
JSON snapshot of the 4-intent + OOS subset used by
tests/data/test_oos_handling.py.

Re-run manually if the upstream dataset shape needs to change:
    uv run python tests/_helpers/extract_clinc150_subset.py

The output file (tests/assets/data/clinc150_oos_input.json) is committed
to the repo so CI never touches HF Hub for it.

The snapshot follows the standard `Dataset.from_json` shape: a single
`train` split containing both the 4-intent labeled samples AND the raw
OOS samples (label=None), plus an `intents` list. This matches how
clinc150 itself stores OOS — they live in `train` alongside the labeled
samples — so the downstream StratifiedSplitter sequence in the test
operates on identical data shapes whether sourced from HF Hub or this
JSON snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

from autointent import Dataset

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "assets" / "data" / "clinc150_oos_input.json"


def main() -> None:
    clinc = Dataset.from_hub("DeepPavlov/clinc150")

    intents_subset = clinc.intents[:4]
    intent_ids = {intent.id for intent in intents_subset}

    labeled = clinc["train"].filter(lambda sample: sample["label"] in intent_ids)
    oos = clinc["train"].filter(lambda sample: sample["label"] is None)

    # DatasetReader (the validation layer behind Dataset.from_json) forbids
    # extra split names, so OOS samples cannot live in their own split — they
    # ride along inside `train` exactly as upstream clinc150 stores them.
    train_samples = labeled.to_list() + oos.to_list()

    payload = {
        "train": train_samples,
        "intents": [intent.model_dump() for intent in intents_subset],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
    print(f"wrote {OUTPUT_PATH}")  # noqa: T201 — one-shot CLI script


if __name__ == "__main__":
    main()
