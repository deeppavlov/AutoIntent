"""Schemes."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypeAlias, TypeVar, Union, get_args, get_origin, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    RootModel,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from autointent.custom_types import NodeType
from autointent.modules.base import BaseModule
from autointent.nodes.info import DecisionNodeInfo, EmbeddingNodeInfo, RegexNodeInfo, ScoringNodeInfo


class ParamSpace(BaseModel, ABC):
    """Base class for search space used in optuna."""

    @abstractmethod
    def n_possible_values(self) -> int | None:
        """Calculate the number of possible values in the search space.

        Returns:
            The number of possible values or None if search space is continuous.
        """


ParamSpaceT = TypeVar("ParamSpaceT", bound=ParamSpace)


class ParamSpaceInt(ParamSpace):
    """Integer parameter search space configuration."""

    low: int = Field(..., description="Lower boundary of the search space.")
    high: int = Field(..., description="Upper boundary of the search space.")
    step: int = Field(1, description="Step size for the search space.")
    log: bool = Field(False, description="Indicates whether to use a logarithmic scale.")

    def n_possible_values(self) -> int:
        """Calculate the number of possible values in the search space.

        Returns:
            The number of possible values.
        """
        return (self.high - self.low) // self.step + 1


class ParamSpaceFloat(ParamSpace):
    """Float parameter search space configuration."""

    low: float = Field(..., description="Lower boundary of the search space.")
    high: float = Field(..., description="Upper boundary of the search space.")
    step: float | None = Field(None, description="Step size for the search space (if applicable).")
    log: bool = Field(False, description="Indicates whether to use a logarithmic scale.")

    @field_validator("step")
    @classmethod
    def validate_step_with_log(cls, v: float | None, info: ValidationInfo) -> float | None:
        """Validate that step is not used when log is True.

        Args:
            v: The step value to validate
            info: Validation info containing other field values

        Returns:
            The validated step value

        Raises:
            ValueError: If step is provided when log is True
        """
        if info.data.get("log", False) and v is not None:
            msg = "Step cannot be used when log is True. See optuna docs on `suggest_float` (https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float)."
            raise ValueError(msg)
        return v

    def n_possible_values(self) -> int | None:
        """Calculate the number of possible values in the search space.

        Returns:
            The number of possible values or None if search space is continuous.
        """
        if self.step is None:
            return None
        return int((self.high - self.low) // self.step) + 1


def unwrap_annotated(tp: type) -> type:
    """Unwrap the Annotated type to get the actual type.

    :param tp: Type to unwrap
    :return: Unwrapped type
    """
    # Check if the type is an Annotated type using get_origin
    # Annotated[int, "some metadata"] would have origin as Annotated
    # If it is Annotated, extract the first argument which is the actual type
    # Otherwise return the original type unchanged
    return get_args(tp)[0] if get_origin(tp) is Annotated else tp


def type_matches(target: type, tp: type) -> bool:
    """Recursively check if the target type is present in the given type.

    This function handles union types by unwrapping Annotated types where necessary.

    :param target: Target type
    :param tp: Given type
    :return: If the target type is present in the given type
    """
    # Get the origin of the type to determine if it's a generic type
    # For example, Union, List, Dict, etc.
    origin = get_origin(tp)

    # If the type is a Union (e.g., int | str or Union[int, str])
    if origin is Union:
        # Check if any of the union's arguments match the target type
        # Recursively call type_matches for each argument in the union
        return any(type_matches(target, arg) for arg in get_args(tp))

    # For non-Union types, unwrap any Annotated wrapper and compare with the target type
    # This handles cases like Annotated[int, "some description"] matching with int
    return unwrap_annotated(tp) is target


def get_optuna_class(param_type: type) -> type[ParamSpaceInt | ParamSpaceFloat] | None:
    """Get the Optuna class for the given parameter type.

    If the (possibly annotated or union) type includes int or float, this function
    returns the corresponding search space class.

    :param param_type: Parameter type (could be a union, annotated type, or container)
    :return: ParamSpaceInt if the type matches int, ParamSpaceFloat if it matches float, else None.
    """
    # Check if the parameter type matches or includes int
    if type_matches(int, param_type):
        return ParamSpaceInt
    # Check if the parameter type matches or includes float
    if type_matches(float, param_type):
        return ParamSpaceFloat
    # Return None if neither int nor float types match
    return None


def generate_models_and_union_type_for_classes(
    classes: list[type[BaseModule]],
) -> tuple[type[BaseModel], dict[str, type[BaseModel]]]:
    """Dynamically generates Pydantic models for class constructors and creates a union type.

    This function takes a list of module classes and creates Pydantic models that represent
    their initialization parameters. It also creates a union type of all these models.

    Args:
        classes: A list of BaseModule subclasses to generate models for

    Returns:
        A tuple containing:
        - A union type of all generated models
        - A dictionary mapping module names to their generated model classes
    """
    # Dictionary to store the generated models, keyed by module name
    models: dict[str, type[BaseModel]] = {}

    # Iterate through each module class
    for cls in classes:
        # Get the signature of the from_context method to extract parameters
        init_signature = inspect.signature(cls.from_context)
        # Get the global namespace for resolving variables in type hints
        globalns = getattr(cls.from_context, "__globals__", {})
        # Get type hints with forward references resolved and extra info preserved
        type_hints = get_type_hints(cls.from_context, globalns, None, include_extras=True)

        # Check if the method accepts arbitrary keyword arguments (**kwargs)
        has_kwarg_arg = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in init_signature.parameters.values())

        # Initialize fields dictionary with common fields for all models
        fields = {
            # Module name field with a Literal type restricting it to this specific class name
            "module_name": (Literal[cls.name], Field(...)),
            # Optional field for number of trials in hyperparameter optimization
            "n_trials": (PositiveInt | None, Field(None, description="Number of trials")),
            # Config field to control extra fields behavior based on kwargs presence
            "model_config": (ConfigDict, ConfigDict(extra="allow" if has_kwarg_arg else "forbid")),
        }

        # Process each parameter from the method signature
        for param_name, param in init_signature.parameters.items():
            # Skip self, cls, context parameters and **kwargs
            if param_name in ("self", "cls", "context") or param.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            # Get the parameter's type annotation, defaulting to Any if not specified
            param_type: TypeAlias = type_hints.get(param_name, Any)  # type: ignore[valid-type]  # noqa: PYI042

            # Create a Field with default value if provided, otherwise make it required
            field = Field(default=[param.default]) if param.default is not inspect.Parameter.empty else Field(...)

            # Check if this parameter should have an Optuna search space
            search_type = get_optuna_class(param_type)

            if search_type is None:
                # Regular parameter: use a list of the parameter's type
                fields[param_name] = (list[param_type], field)
            else:
                # Parameter eligible for optimization: allow either list of values or search space
                fields[param_name] = (list[param_type] | search_type, field)

        # Generate a name for the model class
        model_name = f"{cls.__name__}InitModel"

        # Dynamically create a Pydantic model class for this module
        models[cls.name] = type(
            model_name,
            (BaseModel,),  # Inherit from BaseModel
            {
                # Set type annotations for all fields
                "__annotations__": {k: v[0] for k, v in fields.items()},
                # Set field objects for all fields
                **{k: v[1] for k, v in fields.items()},
            },
        )

    # Return a union type of all models and the dictionary of models
    return Union[tuple(models.values())], models  # type: ignore[return-value]  # noqa: UP007


DecisionSearchSpaceType, DecisionNodesBaseModels = generate_models_and_union_type_for_classes(
    list(DecisionNodeInfo.modules_available.values())
)
DecisionMetrics = Literal[tuple(DecisionNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class DecisionNodeValidator(BaseModel):
    """Search space configuration for the Decision node."""

    node_type: NodeType = NodeType.decision
    target_metric: DecisionMetrics  # type: ignore[valid-type]
    metrics: list[DecisionMetrics] | None = None  # type: ignore[valid-type]
    search_space: list[DecisionSearchSpaceType]  # type: ignore[valid-type]


EmbeddingSearchSpaceType, EmbeddingBaseModels = generate_models_and_union_type_for_classes(
    list(EmbeddingNodeInfo.modules_available.values())
)
EmbeddingMetrics: TypeAlias = Literal[tuple(EmbeddingNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class EmbeddingNodeValidator(BaseModel):
    """Search space configuration for the Embedding node."""

    node_type: NodeType = NodeType.embedding
    target_metric: EmbeddingMetrics
    metrics: list[EmbeddingMetrics] | None = None
    search_space: list[EmbeddingSearchSpaceType]  # type: ignore[valid-type]


ScoringSearchSpaceType, ScoringNodesBaseModels = generate_models_and_union_type_for_classes(
    list(ScoringNodeInfo.modules_available.values())
)
ScoringMetrics: TypeAlias = Literal[tuple(ScoringNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class ScoringNodeValidator(BaseModel):
    """Search space configuration for the Scoring node."""

    node_type: NodeType = NodeType.scoring
    target_metric: ScoringMetrics
    metrics: list[ScoringMetrics] | None = None
    search_space: list[ScoringSearchSpaceType]  # type: ignore[valid-type]


RegexpSearchSpaceType, RegexNodesBaseModels = generate_models_and_union_type_for_classes(
    list(RegexNodeInfo.modules_available.values())
)
RegexpMetrics: TypeAlias = Literal[tuple(RegexNodeInfo.metrics_available.keys())]  # type: ignore[valid-type]


class RegexNodeValidator(BaseModel):
    """Search space configuration for the Regexp node."""

    node_type: NodeType = NodeType.regex
    target_metric: RegexpMetrics
    metrics: list[RegexpMetrics] | None = None
    search_space: list[RegexpSearchSpaceType]  # type: ignore[valid-type]


NodeValidatorType: TypeAlias = (
    EmbeddingNodeValidator | ScoringNodeValidator | DecisionNodeValidator | RegexNodeValidator
)
SearchSpaceType: TypeAlias = (
    DecisionSearchSpaceType | EmbeddingSearchSpaceType | ScoringSearchSpaceType | RegexpSearchSpaceType  # type: ignore[valid-type]
)


class SearchSpaceConfig(RootModel[list[SearchSpaceType]]):
    """Search space configuration."""

    def __iter__(
        self,
    ) -> Iterator[SearchSpaceType]:
        """Iterate over the root."""
        return iter(self.root)

    def __getitem__(self, item: int) -> SearchSpaceType:
        """To get item directly from the root.

        :param item: Index

        :return: Item
        """
        return self.root[item]

    @model_validator(mode="before")
    @classmethod
    def validate_nodes(cls, data: list[Any]) -> list[Any]:  # noqa: C901
        """Validate the search space configuration.

        Args:
            data: List of search space configurations.

        Returns:
            List of validated search space configurations.
        """
        error_message = ""
        for i, item in enumerate(data):
            if isinstance(item, BaseModel):
                continue
            if not isinstance(item, dict):
                msg = "Each search space configuration must be a dictionary."
                raise TypeError(msg)
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


class OptimizationSearchSpaceConfig(RootModel[list[NodeValidatorType]]):
    """Optimizer configuration."""

    def __iter__(
        self,
    ) -> Iterator[NodeValidatorType]:
        """Iterate over the root."""
        return iter(self.root)

    def __getitem__(self, item: int) -> NodeValidatorType:
        """To get item directly from the root.

        :param item: Index

        :return: Item
        """
        return self.root[item]

    @model_validator(mode="before")
    @classmethod
    def validate_nodes(cls, data: list[Any]) -> list[Any]:  # noqa: PLR0912,C901
        """Validate the search space configuration.

        Args:
            data: List of search space configurations.

        Returns:
            List of validated search space configurations.
        """
        error_message = ""
        for i, item in enumerate(data):
            if isinstance(item, BaseModel):
                continue
            if not isinstance(item, dict):
                msg = "Each search space configuration must be a dictionary."
                raise TypeError(msg)
            if "node_type" not in item:
                msg = "Each search space configuration must have a 'node_type' key."
                raise TypeError(msg)
            if not isinstance(item.get("search_space"), list):
                msg = "Each search space configuration must have a 'search_space' key of type list."
                raise TypeError(msg)
            for search_space in item["search_space"]:
                node_name = search_space.get("module_name")
                if node_name is None:
                    error_message += f"Search space configuration at index {i} is missing 'module_name'.\n"
                    continue
                if item["node_type"] == NodeType.decision.value:
                    node_class = DecisionNodesBaseModels[node_name]
                elif item["node_type"] == NodeType.embedding.value:
                    node_class = EmbeddingBaseModels[node_name]
                elif item["node_type"] == NodeType.scoring.value:
                    node_class = ScoringNodesBaseModels[node_name]
                elif item["node_type"] == NodeType.regex.value:
                    node_class = RegexNodesBaseModels[node_name]
                else:
                    error_message += f"Unknown node type '{item['node_type']}' at index {i}.\n"
                    break

                try:
                    node_class(**search_space)
                except ValidationError as e:
                    error_message += f"Search space configuration at index {i} {node_name} is invalid: {e}\n"
                    continue
        if len(error_message) > 0:
            raise TypeError(error_message)
        return data
