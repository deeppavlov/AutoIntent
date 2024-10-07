from dataclasses import fields
from typing import Any

from pydantic import BaseModel, create_model

from .base import ModuleConfig


class SearchSpace(BaseModel):
    module_type: str


def create_search_space_config(data_class: type[ModuleConfig]) -> type[SearchSpace]:
    data_class_fields = fields(data_class)
    new_fields: dict[str, Any] = {}

    for field in data_class_fields:
        if field.name == "module_type":
            new_fields[field.name] = (list[field.type], field.default)
        elif field.name != "_target_":
            new_fields[field.name] = (list[field.type], ...)

    model_name = data_class.__name__ + "SearchSpace"
    return create_model(model_name, **new_fields)
