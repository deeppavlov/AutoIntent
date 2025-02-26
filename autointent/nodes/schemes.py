"""Schemes."""

import functools
import inspect
import operator
from collections.abc import Iterator
from types import NoneType, UnionType
from typing import Annotated, Any, Literal, TypeAlias, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, RootModel

from autointent.custom_types import NodeType, ParamSpaceFloat, ParamSpaceInt
from autointent.modules.abc import BaseModule
from autointent.nodes.info import DecisionNodeInfo, EmbeddingNodeInfo, RegexNodeInfo, ScoringNodeInfo


def unwrap_annotated(tp: type) -> type:
    """
    Unwrap the Annotated type to get the actual type.

    :param tp: Type to unwrap
    :return: Unwrapped type
    """
    return get_args(tp)[0] if get_origin(tp) is Annotated else tp


def type_matches(target: type, tp: type) -> bool:
    """
    Recursively check if the target type is present in the given type.

    This function handles union types and generic types (e.g. dict[...] by checking
    their origin) after unwrapping Annotated types.

    :param target: Target type to check for.
    :param tp: Given type which may be a union, generic, or annotated type.
    :return: True if the target type is present in the given type.
    """
    origin = get_origin(tp)
    if origin is Union:
        return any(type_matches(target, arg) for arg in get_args(tp))

    # Unwrap Annotated types, if any.
    unwrapped = unwrap_annotated(tp)

    # If the unwrapped type is a generic type, check its origin.
    generic_origin = get_origin(unwrapped)
    if generic_origin is not None:
        return generic_origin is target

    return unwrapped is target


def get_optuna_class(param_type: type) -> type[ParamSpaceInt | ParamSpaceFloat] | None:
    """
    Get the Optuna class for the given parameter type.

    If the (possibly annotated or union) type includes int or float, this function
    returns the corresponding search space class.

    :param param_type: Parameter type (could be a union, annotated type, or container)
    :return: ParamSpaceInt if the type matches int, ParamSpaceFloat if it matches float, else None.
    """
    if type_matches(int, param_type):
        return ParamSpaceInt
    if type_matches(float, param_type):
        return ParamSpaceFloat
    return None


def to_union(types: list[type]) -> type:
    """Convert a tuple of types into a union type."""
    return functools.reduce(operator.or_, types)


def generate_models_and_union_type_for_classes(  # noqa: PLR0912, C901
    classes: list[type[BaseModule]],
) -> type[BaseModel]:
    """Dynamically generates Pydantic models for class constructors and creates a union type."""
    models: dict[str, type[BaseModel]] = {}

    for cls in classes:
        init_signature = inspect.signature(cls.from_context)
        globalns = getattr(cls.from_context, "__globals__", {})
        type_hints = get_type_hints(cls.from_context, globalns, None, include_extras=True)  # Resolve forward refs

        fields = {
            "module_name": (Literal[cls.name], Field(...)),
            "n_trials": (PositiveInt | None, Field(None, description="Number of trials")),
            "model_config": (ConfigDict, ConfigDict(extra="forbid")),
        }

        for param_name, param in init_signature.parameters.items():
            if param_name in ("self", "cls", "context"):
                continue

            param_type: TypeAlias = type_hints.get(param_name, Any)  # type: ignore[valid-type]  # noqa: PYI042
            field = Field(default=[param.default]) if param.default is not inspect.Parameter.empty else Field(...)
            if not type_matches(dict, param_type):
                search_type = get_optuna_class(param_type)
                if search_type is None:
                    fields[param_name] = (list[param_type], field)
                else:
                    fields[param_name] = (list[param_type] | search_type, field)
            else:
                dict_key_type, dict_values_types = get_args(param_type)
                is_optional = False
                if dict_values_types is NoneType:  # if dict is optional
                    is_optional = True
                    dict_key_type, dict_values_types = get_args(dict_key_type)
                if get_origin(dict_values_types) is UnionType:
                    filed_types: list[type[Any]] = []
                    for value in get_args(dict_values_types):
                        search_type = get_optuna_class(value)
                        if search_type is not None:
                            filed_types.append(search_type)
                        filed_types.append(list[value])  # type: ignore[valid-type]
                    filed_type = to_union(filed_types)
                else:
                    filed_type = dict_values_types

                if is_optional:
                    fields[param_name] = (dict[dict_key_type, filed_type] | None, field)  # type: ignore[valid-type]
                else:
                    fields[param_name] = (dict[dict_key_type, filed_type], field)  # type: ignore[valid-type]

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


RegexpSearchSpaceType: TypeAlias = generate_models_and_union_type_for_classes(  # type: ignore[valid-type]
    list(RegexNodeInfo.modules_available.values())
)
RegexpMetrics: TypeAlias = Literal[tuple(RegexNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class RegexNodeValidator(BaseModel):
    """Search space configuration for the Regexp node."""

    node_type: NodeType = NodeType.regex
    target_metric: RegexpMetrics
    metrics: list[RegexpMetrics] | None = None
    search_space: list[RegexpSearchSpaceType]


SearchSpaceTypes: TypeAlias = ScoringNodeValidator | EmbeddingNodeValidator | DecisionNodeValidator | RegexNodeValidator


class OptimizationSearchSpaceConfig(RootModel[list[SearchSpaceTypes]]):
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
