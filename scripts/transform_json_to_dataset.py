from datasets import Dataset, load_dataset, DatasetDict


def transform_dataset(
    path: str,
) -> tuple[Dataset | None, Dataset | None, Dataset | None]:
    ds: DatasetDict = load_dataset("json", data_files=path)["train"]
    utterance_ds = None
    tags_ds = None
    intents_ds = None
    if "utterances" in ds.column_names:
        utterance_ds = Dataset.from_list(ds["utterances"][0])
    if "tags" in ds.column_names:
        tags_ds = Dataset.from_list(ds["tags"][0])
    if "intents" in ds.column_names:
        intents_ds = Dataset.from_list(ds["intents"][0])
    return utterance_ds, tags_ds, intents_ds


def push_json_to_hub(path: str, ds_name: str) -> None:
    utterance_ds, tags_ds, intents_ds = transform_dataset(path)
    if utterance_ds is not None:
        utterance_ds.push_to_hub(ds_name, config_name="utterances")
    if tags_ds is not None:
        tags_ds.push_to_hub(ds_name, config_name="tags")
    if intents_ds is not None:
        intents_ds.push_to_hub(ds_name, config_name="intents")


if __name__ == "__main__":
    push_json_to_hub("../tests/assets/data/clinc_subset_multilabel.json", "clinc_subset_multilabel")
