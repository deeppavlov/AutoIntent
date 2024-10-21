from collections.abc import Mapping
from typing import ClassVar

from autointent.configs.modules import SCORING_MODULES_CONFIGS, ModuleConfig
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL, ScoringMetricFn
from autointent.modules import SCORING_MODULES_MULTICLASS, SCORING_MODULES_MULTILABEL, ScoringModule

from .base import NodeInfo


class ScoringNodeInfo(NodeInfo):
    metrics_available: ClassVar[Mapping[str, ScoringMetricFn]] = SCORING_METRICS_MULTICLASS | SCORING_METRICS_MULTILABEL

    modules_available: ClassVar[Mapping[str, type[ScoringModule]]] = (
        SCORING_MODULES_MULTICLASS | SCORING_MODULES_MULTILABEL
    )

    modules_configs: ClassVar[Mapping[str, type[ModuleConfig]]] = SCORING_MODULES_CONFIGS

    node_type = "scoring"
