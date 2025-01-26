"""Schemes."""

import inspect
from typing import Any, Literal, TypeAlias, Union, get_type_hints

from pydantic import BaseModel, Field

from autointent.custom_types import NodeType
from autointent.modules.abc import Module
from autointent.nodes import EmbeddingNodeInfo, ScoringNodeInfo


def generate_models_and_union_type_for_classes(
    classes: list[type[Module]],
) -> type[BaseModel]:
    """Dynamically generates Pydantic models for each class constructor (`__init__`) and a union type for all models.

    :param classes: A list of classes to generate models from.

    :return: A union type of all models.
    """
    models: dict[str, type[BaseModel]] = {}

    for cls in classes:
        init_signature = inspect.signature(cls.__init__)
        type_hints = get_type_hints(cls.__init__)

        fields: dict[str, Any] = {
            "module_name": cls.name,
        }
        for param_name, param in init_signature.parameters.items():
            if param_name == "self":  # Skip `self` in class methods
                continue
            param_type = type_hints.get(param_name, Any)
            default = param.default if param.default is not inspect.Parameter.empty else ...
            fields[param_name] = (param_type, Field(default=default))

        # Dynamically create a Pydantic model for each class's __init__
        model_name = f"{cls.__name__}InitModel"
        models[cls.__name__] = type(
            model_name,
            (BaseModel,),
            {
                "__annotations__": {k: v[0] for k, v in fields.items()},
                **{k: v[1] for k, v in fields.items()},
            },
        )

    # Create a union type of all models
    return Union[tuple(models.values())]  # type: ignore[return-value] # noqa: UP007


DecisionSearchSpaceType: TypeAlias = generate_models_and_union_type_for_classes(  # type: ignore[valid-type]
    list(ScoringNodeInfo.modules_available.values())
)
DecisionMetrics: TypeAlias = Literal[tuple(ScoringNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class SearchSpaceDecisionConfigNodeItem(BaseModel):
    """Search space configuration for the Decision node."""

    node_type: NodeType = NodeType.decision
    metric: DecisionMetrics
    search_space: DecisionSearchSpaceType


EmbeddingSearchSpaceType: TypeAlias = generate_models_and_union_type_for_classes(  # type: ignore[valid-type]
    list(EmbeddingNodeInfo.modules_available.values())
)
EmbeddingMetrics: TypeAlias = Literal[tuple(EmbeddingNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class SearchSpaceEmbeddingConfigNodeItem(BaseModel):
    """Search space configuration for the Embedding node."""

    node_type: NodeType = NodeType.embedding
    metric: EmbeddingMetrics
    search_space: EmbeddingSearchSpaceType


ScoringSearchSpaceType: TypeAlias = generate_models_and_union_type_for_classes(  # type: ignore[valid-type]
    list(ScoringNodeInfo.modules_available.values())
)
ScoringMetrics: TypeAlias = Literal[tuple(ScoringNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class SearchSpaceScoringConfigNodeItem(BaseModel):
    """Search space configuration for the Scoring node."""

    node_type: NodeType = NodeType.scoring
    metric: ScoringMetrics
    search_space: ScoringSearchSpaceType


RegexpSearchSpaceType: TypeAlias = generate_models_and_union_type_for_classes(  # type: ignore[valid-type]
    list(ScoringNodeInfo.modules_available.values())
)
RegexpMetrics: TypeAlias = Literal[tuple(ScoringNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class SearchSpaceRegexpConfigNodeItem(BaseModel):
    """Search space configuration for the Regexp node."""

    node_type: NodeType = NodeType.regexp
    metric: RegexpMetrics
    search_space: RegexpSearchSpaceType
