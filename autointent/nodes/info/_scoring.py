"""Scoring node info."""

from collections.abc import Mapping
from typing import ClassVar

from autointent.custom_types import NodeType
from autointent.metrics import SCORING_METRICS_MULTICLASS, SCORING_METRICS_MULTILABEL, ScoringMetricFn
from autointent.modules import SCORING_MODULES
from autointent.modules.base import BaseScorer

from ._base import NodeInfo


class ScoringNodeInfo(NodeInfo):
    """Scoring node info."""

    metrics_available: ClassVar[Mapping[str, ScoringMetricFn]] = SCORING_METRICS_MULTICLASS | SCORING_METRICS_MULTILABEL

    modules_available: ClassVar[Mapping[str, type[BaseScorer]]] = SCORING_MODULES

    node_type = NodeType.scoring

    multiclass_available_metrics: ClassVar[Mapping[str, ScoringMetricFn]] = SCORING_METRICS_MULTICLASS
    multilabel_available_metrics: ClassVar[Mapping[str, ScoringMetricFn]] = SCORING_METRICS_MULTILABEL
