"""Module for handling tags in datasets.

This module provides functionality to manage and collect tags associated with
intent classes in a dataset.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from ._schemas import Dataset


@dataclass
class Tag:
    """
    Represents a tag associated with intent classes.

    Tags are used to define constraints such that if two intent classes share
    a common tag, they cannot both be assigned to the same sample.
    """

    tag_name: str
    intent_ids: list[int] = field(default_factory=list)


def collect_tags(dataset: Dataset) -> list[Tag]:
    """
    Collect tags from a dataset and map them to their associated intent IDs.

    :param dataset: A `Dataset` object containing intents and their tags.
    :return: A list of `Tag` objects, each representing a tag and the intent IDs associated with it.
    """
    tag_mapping = defaultdict(list)
    for intent in dataset.intents:
        for tag in intent.tags:
            tag_mapping[tag].append(intent.id)

    return [Tag(tag_name=tag, intent_ids=intent_ids) for tag, intent_ids in tag_mapping.items()]
