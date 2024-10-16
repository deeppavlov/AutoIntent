from collections.abc import Mapping
from typing import ClassVar

from autointent.configs.modules import ModuleConfig
from autointent.metrics import PredictionMetricFn, RetrievalMetricFn, ScoringMetricFn
from autointent.metrics.regexp import RegexpMetricFn
from autointent.modules import Module

METRIC_FN = PredictionMetricFn | RegexpMetricFn | RetrievalMetricFn | ScoringMetricFn


class NodeInfo:
    metrics_available: ClassVar[Mapping[str, METRIC_FN]]
    modules_available: ClassVar[Mapping[str, type[Module]]]
    modules_configs: ClassVar[Mapping[str, type[ModuleConfig]]]
    node_type: str
