from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from autointent import Dataset
from autointent.context.data_handler._stratification import StratifiedSplitter


def count_oos(split: HFDataset):
    return len(split.filter(lambda sample: sample["label"] is None))


def create_clinc150_subset():
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

    return Dataset.from_dict(
        {"train_0": train_0, "train_1": train_1, "validation_0": val_0, "validation_1": val_1, "test": test}
    )


def test():
    dataset = create_clinc150_subset()

    desired_specs = {
        "test": {"total": 12, "oos": 4},
        "train_0": {"total": 18, "oos": 0},
        "train_1": {"total": 18, "oos": 12},
        "validation_0": {"total": 4, "oos": 0},
        "validation_1": {"total": 8, "oos": 4},
    }

    for split_name in dataset:
        assert len(dataset[split_name]) == desired_specs[split_name]["total"]
        assert count_oos(dataset[split_name]) == desired_specs[split_name]["oos"]
