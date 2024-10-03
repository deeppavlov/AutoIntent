from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Tag:
    """
    If two intent classes have common tag they can't be both assigned to one sample
    """

    tag_name: str
    intent_ids: list[int] = field(default_factory=list)  # classes with this tag


def collect_tags(intent_records: list[dict]) -> list[Tag]:
    tagwise_intent_ids = defaultdict(list)
    for dct in intent_records:
        if "tags" not in dct:
            continue
        for tag_name in dct["tags"]:
            tagwise_intent_ids[tag_name].append(dct["intent_id"])
    return [Tag(tag_name, intent_ids) for tag_name, intent_ids in tagwise_intent_ids.items()]
