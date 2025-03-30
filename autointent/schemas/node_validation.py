"""Schemes."""

import inspect
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypeAlias, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, RootModel, ValidationError, model_validator

from autointent.custom_types import NodeType
from autointent.modules import BaseModule
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

    This function handles union types by unwrapping Annotated types where necessary.

    :param target: Target type
    :param tp: Given type
    :return: If the target type is present in the given type
    """
    origin = get_origin(tp)

    if origin is Union:  # float | list[float]
        return any(type_matches(target, arg) for arg in get_args(tp))
    return unwrap_annotated(tp) is target


class ParamSpaceInt(BaseModel):
    """Integer parameter search space configuration."""

    low: int = Field(..., description="Lower boundary of the search space.")
    high: int = Field(..., description="Upper boundary of the search space.")
    step: int = Field(1, description="Step size for the search space.")
    log: bool = Field(False, description="Indicates whether to use a logarithmic scale.")


class ParamSpaceFloat(BaseModel):
    """Float parameter search space configuration."""

    low: float = Field(..., description="Lower boundary of the search space.")
    high: float = Field(..., description="Upper boundary of the search space.")
    step: float | None = Field(None, description="Step size for the search space (if applicable).")
    log: bool = Field(False, description="Indicates whether to use a logarithmic scale.")


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


def generate_models_and_union_type_for_classes(
    classes: list[type[BaseModule]],
) -> tuple[type[BaseModel], dict[str, type[BaseModel]]]:
    """Dynamically generates Pydantic models for class constructors and creates a union type."""
    models: dict[str, type[BaseModel]] = {}

    for cls in classes:
        init_signature = inspect.signature(cls.from_context)
        globalns = getattr(cls.from_context, "__globals__", {})
        type_hints = get_type_hints(cls.from_context, globalns, None, include_extras=True)  # Resolve forward refs

        has_kwarg_arg = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in init_signature.parameters.values()
        )

        fields = {
            "module_name": (Literal[cls.name], Field(...)),
            "n_trials": (PositiveInt | None, Field(None, description="Number of trials")),
            "model_config": (ConfigDict, ConfigDict(extra="allow" if has_kwarg_arg else "forbid")),
        }

        for param_name, param in init_signature.parameters.items():
            # skip self, cls, context, and **kwargs
            if param_name in ("self", "cls", "context") or param.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            param_type: TypeAlias = type_hints.get(param_name, Any)  # type: ignore[valid-type]  # noqa: PYI042
            field = Field(default=[param.default]) if param.default is not inspect.Parameter.empty else Field(...)
            search_type = get_optuna_class(param_type)
            if search_type is None:
                fields[param_name] = (list[param_type], field)
            else:
                fields[param_name] = (list[param_type] | search_type, field)

        model_name = f"{cls.__name__}InitModel"
        models[cls.name] = type(
            model_name,
            (BaseModel,),
            {
                "__annotations__": {k: v[0] for k, v in fields.items()},
                **{k: v[1] for k, v in fields.items()},
            },
        )

    return Union[tuple(models.values())], models


DecisionSearchSpaceType, DecisionNodesBaseModels = generate_models_and_union_type_for_classes(  # type: ignore[valid-type]
    list(DecisionNodeInfo.modules_available.values())
)
DecisionMetrics = Literal[tuple(DecisionNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class DecisionNodeValidator(BaseModel):
    """Search space configuration for the Decision node."""

    node_type: NodeType = NodeType.decision
    target_metric: DecisionMetrics
    metrics: list[DecisionMetrics] | None = None
    search_space: list[DecisionSearchSpaceType]


EmbeddingSearchSpaceType, EmbeddingBaseModels = generate_models_and_union_type_for_classes(
    list(EmbeddingNodeInfo.modules_available.values())
)
EmbeddingMetrics: TypeAlias = Literal[tuple(EmbeddingNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class EmbeddingNodeValidator(BaseModel):
    """Search space configuration for the Embedding node."""

    node_type: NodeType = NodeType.embedding
    target_metric: EmbeddingMetrics
    metrics: list[EmbeddingMetrics] | None = None
    search_space: list[EmbeddingSearchSpaceType]


ScoringSearchSpaceType, ScoringNodesBaseModels = generate_models_and_union_type_for_classes(
    list(ScoringNodeInfo.modules_available.values())
)
ScoringMetrics: TypeAlias = Literal[tuple(ScoringNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class ScoringNodeValidator(BaseModel):
    """Search space configuration for the Scoring node."""

    node_type: NodeType = NodeType.scoring
    target_metric: ScoringMetrics
    metrics: list[ScoringMetrics] | None = None
    search_space: list[ScoringSearchSpaceType]


RegexpSearchSpaceType, RegexNodesBaseModels = generate_models_and_union_type_for_classes(
    list(RegexNodeInfo.modules_available.values())
)
RegexpMetrics: TypeAlias = Literal[tuple(RegexNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class RegexNodeValidator(BaseModel):
    """Search space configuration for the Regexp node."""

    node_type: NodeType = NodeType.regex
    target_metric: RegexpMetrics
    metrics: list[RegexpMetrics] | None = None
    search_space: list[RegexpSearchSpaceType]


SearchSpaceTypes: TypeAlias = EmbeddingNodeValidator | ScoringNodeValidator | DecisionNodeValidator | RegexNodeValidator


class SearchSpaceConfig(RootModel[list[DecisionSearchSpaceType | EmbeddingSearchSpaceType | ScoringSearchSpaceType | RegexpSearchSpaceType]]):
    """Search space configuration."""

    def __iter__(
        self,
    ) -> Iterator[DecisionSearchSpaceType | EmbeddingSearchSpaceType | ScoringSearchSpaceType | RegexpSearchSpaceType]:
        """Iterate over the root."""
        return iter(self.root)

    def __getitem__(
        self, item: int
    ) -> DecisionSearchSpaceType | EmbeddingSearchSpaceType | ScoringSearchSpaceType | RegexpSearchSpaceType:
        """
        To get item directly from the root.

        :param item: Index

        :return: Item
        """
        return self.root[item]

    @model_validator(mode='before')
    @classmethod
    def validate_nodes(cls, data: list[Any]) -> list[Any]:
        if not isinstance(data, list):
            raise TypeError("The root must be a list of search space configurations.")
        error_message = ""
        for i, item in enumerate(data):
            if isinstance(item, BaseModel):
                continue
            if not isinstance(item, dict):
                raise TypeError("Each search space configuration must be a dictionary.")
            node_name = item.get("module_name")
            if node_name is None:
                error_message += f"Search space configuration at index {i} is missing 'module_name'.\n"
                continue

            if node_name in DecisionNodesBaseModels:
                node_class = DecisionNodesBaseModels[node_name]
            elif node_name in EmbeddingBaseModels:
                node_class = EmbeddingBaseModels[node_name]
            elif node_name in ScoringNodesBaseModels:
                node_class = ScoringNodesBaseModels[node_name]
            elif node_name in RegexNodesBaseModels:
                node_class = RegexNodesBaseModels[node_name]
            else:
                error_message += f"Unknown node type '{item['node_type']}' at index {i}.\n"
                break
            try:
                node_class(**item)
            except ValidationError as e:
                error_message += f"Search space configuration at index {i} {node_name} is invalid: {e}\n"
                continue
        if len(error_message) > 0:
            raise TypeError(error_message)
        return data


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


    @model_validator(mode='before')
    @classmethod
    def validate_nodes(cls, data: list[Any]) -> list[Any]:
        if not isinstance(data, list):
            raise ValueError("The root must be a list of search space configurations.")
        error_message = ""
        for i, item in enumerate(data):
            if isinstance(item, BaseModel):
                continue
            if not isinstance(item, dict):
                raise ValueError("Each search space configuration must be a dictionary.")
            if "node_type" not in item:
                raise ValueError("Each search space configuration must have a 'node_type' key.")
            if not isinstance(item.get("search_space"), list):
                raise ValueError("Each search space configuration must have a 'search_space' key of type list.")
            for search_space in item["search_space"]:
                node_name = search_space.get("module_name")
                if node_name is None:
                    error_message += f"Search space configuration at index {i} is missing 'module_name'.\n"
                    continue
                if item["node_type"] == NodeType.decision.value:
                    node_class = DecisionNodesBaseModels.get(node_name)
                elif item["node_type"] == NodeType.embedding.value:
                    node_class = EmbeddingBaseModels.get(node_name)
                elif item["node_type"] == NodeType.scoring.value:
                    node_class = ScoringNodesBaseModels.get(node_name)
                elif item["node_type"] == NodeType.regex.value:
                    node_class = RegexNodesBaseModels.get(node_name)
                else:
                    error_message += f"Unknown node type '{item['node_type']}' at index {i}.\n"
                    break

                try:
                    node_class(**search_space)
                except ValidationError as e:
                    error_message += f"Search space configuration at index {i} {node_name} is invalid: {e}\n"
                    continue
        if len(error_message) > 0:
            raise ValueError(error_message)
        return data




