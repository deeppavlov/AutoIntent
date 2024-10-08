from dataclasses import dataclass, field, fields, make_dataclass
from typing import Any

from pydantic import BaseModel, create_model

from .base import ModuleConfig


@dataclass
class SearchSpaceDataclass:
    module_type: str


class SearchSpaceModel(BaseModel):
    module_type: str


def create_search_space_dataclass(data_class: type[ModuleConfig]) -> type[SearchSpaceDataclass]:
    data_class_fields = fields(data_class)
    new_fields = []

    for field_ in data_class_fields:
        if field_.name == "module_type":
            new_fields.append((field_.name, field_.type, field(default=field_.default)))
        elif field_.name != "_target_":
            new_fields.append((field_.name, list[field_.type], field(default_factory=list)))

    return make_dataclass(f"{data_class.__name__}SearchSpace", new_fields, bases=(SearchSpaceDataclass,))


def create_search_space_model(data_class: type[ModuleConfig]) -> type[SearchSpaceModel]:
    data_class_fields = fields(data_class)
    new_fields: dict[str, Any] = {}

    for field_ in data_class_fields:
        if field_.name == "module_type":
            new_fields[field_.name] = (field_.type, field_.default)
        elif field_.name != "_target_":
            new_fields[field_.name] = (list[field_.type], field_.default)

    model_name = data_class.__name__ + "SearchSpace"
    return create_model(model_name, **new_fields, __base__=SearchSpaceModel)
