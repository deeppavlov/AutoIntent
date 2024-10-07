from dataclasses import dataclass, field, make_dataclass
from typing import TypeVar, get_type_hints


@dataclass
class SearchSpace:
    module_type: str


T = TypeVar("T")


def create_search_space_config(original_cls: type[T], module_type: str) -> type[T]:
    # Get the type hints of the original class
    type_hints = get_type_hints(original_cls)

    # Create a new dictionary for the fields of the new class
    new_fields = []

    # Iterate over the fields of the original class
    for field_name, field_type in type_hints.items():
        if field_name != "_target_":
            # Change the type annotation to a list of the original type
            new_fields.append(field_name, list[field_type], field(default_factory=list))

    new_fields.append("module_type", str, module_type)

    # Create the new data class
    return make_dataclass(f"{original_cls.__name__}SearchSpace", new_fields, bases=(SearchSpace,))
