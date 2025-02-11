"""Schemes."""

import inspect
from collections.abc import Iterator
from typing import Any, Literal, TypeAlias, Union, get_type_hints

from pydantic import BaseModel, Field, RootModel

from autointent.custom_types import NodeType
from autointent.modules.abc import Module
from autointent.nodes import DecisionNodeInfo, EmbeddingNodeInfo, ScoringNodeInfo


def generate_models_and_union_type_for_classes(
    classes: list[type[Module]],
) -> type[BaseModel]:
    """Dynamically generates Pydantic models for class constructors and creates a union type."""
    models: dict[str, type[BaseModel]] = {}

    for cls in classes:
        init_signature = inspect.signature(cls.from_context)
        globalns = getattr(cls.from_context, "__globals__", {})
        type_hints = get_type_hints(cls.from_context, globalns, None)  # Resolve forward refs

        fields = {"module_name": (Literal[cls.name], Field(...))}

        for param_name, param in init_signature.parameters.items():
            if param_name in ("self", "cls", "context"):
                continue

            param_type: TypeAlias = type_hints.get(param_name, Any)  # type: ignore[valid-type]  # noqa: PYI042
            field = Field(default=[param.default]) if param.default is not inspect.Parameter.empty else Field(...)

            fields[param_name] = (list[param_type], field)  # type: ignore[assignment]

        model_name = f"{cls.__name__}InitModel"
        models[cls.__name__] = type(
            model_name,
            (BaseModel,),
            {
                "__annotations__": {k: v[0] for k, v in fields.items()},
                **{k: v[1] for k, v in fields.items()},
            },
        )

    return Union[tuple(models.values())]  # type: ignore[return-value]  # noqa: UP007


DecisionSearchSpaceType: TypeAlias = generate_models_and_union_type_for_classes(  # type: ignore[valid-type]
    list(DecisionNodeInfo.modules_available.values())
)
DecisionMetrics: TypeAlias = Literal[tuple(DecisionNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class DecisionNodeValidator(BaseModel):
    """Search space configuration for the Decision node."""

    node_type: NodeType = NodeType.decision
    target_metric: DecisionMetrics
    metrics: list[DecisionMetrics] | None = None
    search_space: list[DecisionSearchSpaceType]


EmbeddingSearchSpaceType: TypeAlias = generate_models_and_union_type_for_classes(  # type: ignore[valid-type]
    list(EmbeddingNodeInfo.modules_available.values())
)
EmbeddingMetrics: TypeAlias = Literal[tuple(EmbeddingNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class EmbeddingNodeValidator(BaseModel):
    """Search space configuration for the Embedding node."""

    node_type: NodeType = NodeType.embedding
    target_metric: EmbeddingMetrics
    metrics: list[EmbeddingMetrics] | None = None
    search_space: list[EmbeddingSearchSpaceType]


ScoringSearchSpaceType: TypeAlias = generate_models_and_union_type_for_classes(  # type: ignore[valid-type]
    list(ScoringNodeInfo.modules_available.values())
)
ScoringMetrics: TypeAlias = Literal[tuple(ScoringNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class ScoringNodeValidator(BaseModel):
    """Search space configuration for the Scoring node."""

    node_type: NodeType = NodeType.scoring
    target_metric: ScoringMetrics
    metrics: list[ScoringMetrics] | None = None
    search_space: list[ScoringSearchSpaceType]


SearchSpaceTypes: TypeAlias = EmbeddingNodeValidator | ScoringNodeValidator | DecisionNodeValidator


class OptimizationConfig(RootModel[list[SearchSpaceTypes]]):
    """Optimizer configuration."""

    def __iter__(
        self,
    ) -> Iterator[SearchSpaceTypes]:
        """Iterate over the root."""
        return iter(self.root)

    def __getitem__(self, item: int) -> SearchSpaceTypes:
        """
        To get item directly from the root.

        :param item: Index

        :return: Item
        """
        return self.root[item]
