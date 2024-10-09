from collections.abc import Callable
from typing import ClassVar

from autointent.configs.modules import SCORING_MODULES_CONFIGS
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL
from autointent.modules import SCORING_MODULES_MULTICLASS, SCORING_MODULES_MULTILABEL, Module

from .base import NodeInfo


class ScoringNodeInfo(NodeInfo):
    metrics_available: ClassVar[dict[str, Callable]] = SCORING_METRICS_MULTICLASS | SCORING_METRICS_MULTILABEL

    modules_available: ClassVar[dict[str, type[Module]]] = SCORING_MODULES_MULTICLASS | SCORING_MODULES_MULTILABEL

    modules_configs: ClassVar[dict[str, type]] = SCORING_MODULES_CONFIGS

    node_type = "scoring"
