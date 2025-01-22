"""Script for creating clinc150 subset that is used as test data in out testing workflow."""

from datasets import concatenate_datasets

from autointent import Dataset
from autointent.context.data_handler._stratification import StratifiedSplitter

if __name__ == "__main__":
    clinc = Dataset.from_hub("AutoIntent/clinc150")
    intents_subset = clinc.intents[:4]
    intent_ids = [intent.id for intent in intents_subset]
    train = clinc["train"].filter(lambda sample: sample["label"] in intent_ids)
    oos = clinc["train"].filter(lambda sample: sample["label"] is None)

    splitter = StratifiedSplitter(test_size=0.1, label_feature=clinc.label_feature, random_seed=42, shuffle=True)
    _, train = splitter(train, multilabel=False)

    splitter = StratifiedSplitter(test_size=0.1, label_feature=clinc.label_feature, random_seed=42, shuffle=True)
    _, oos = splitter(oos, multilabel=False, allow_oos_in_train=True)

    subset = concatenate_datasets([train, oos])
    splitter = StratifiedSplitter(test_size=0.4, label_feature=clinc.label_feature, random_seed=42, shuffle=True)
    train, val = splitter(subset, multilabel=False, allow_oos_in_train=True)

    splitter = StratifiedSplitter(test_size=0.5, label_feature=clinc.label_feature, random_seed=42, shuffle=True)
    train_0, train_1 = splitter(train, multilabel=False, allow_oos_in_train=False)

    splitter = StratifiedSplitter(test_size=0.5, label_feature=clinc.label_feature, random_seed=42, shuffle=True)
    val, test = splitter(val, multilabel=False, allow_oos_in_train=True)

    splitter = StratifiedSplitter(test_size=0.6, label_feature=clinc.label_feature, random_seed=42, shuffle=True)
    val_0, val_1 = splitter(val, multilabel=False, allow_oos_in_train=False)

    clinc150_subset = Dataset.from_dict(
        {"train_0": train_0, "train_1": train_1, "validation_0": val_0, "validation_1": val_1, "test": test}
    )
    clinc150_subset.to_json("tests/assets/data/clinc_subset.json")
