from collections import defaultdict
from dataclasses import dataclass, field

from .schemas import Dataset


@dataclass
class Tag:
    """
    If two intent classes have common tag they can't be both assigned to one sample
    """

    tag_name: str
    intent_ids: list[int] = field(default_factory=list)  # classes with this tag


def collect_tags(dataset: Dataset) -> list[Tag]:
    tag_mapping = defaultdict(list)
    for intent in dataset.intents:
        for tag in intent.tags:
            tag_mapping[tag].append(intent.id)

    return [Tag(tag_name=tag, intent_ids=intent_ids) for tag, intent_ids in tag_mapping.items()]
