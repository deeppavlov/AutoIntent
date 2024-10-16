from collections.abc import Mapping
from typing import ClassVar

from autointent.configs.modules import PREDICTION_MODULES_CONFIGS, ModuleConfig
from autointent.metrics import PREDICTION_METRICS_MULTICLASS, PREDICTION_METRICS_MULTILABEL, PredictionMetricFn
from autointent.modules import PREDICTION_MODULES_MULTICLASS, PREDICTION_MODULES_MULTILABEL, Module

from .base import NodeInfo


class PredictionNodeInfo(NodeInfo):
    metrics_available: ClassVar[Mapping[str, PredictionMetricFn]] = (
        PREDICTION_METRICS_MULTICLASS | PREDICTION_METRICS_MULTILABEL
    )

    modules_available: ClassVar[Mapping[str, type[Module]]] = (
        PREDICTION_MODULES_MULTICLASS | PREDICTION_MODULES_MULTILABEL
    )

    modules_configs: ClassVar[Mapping[str, type[ModuleConfig]]] = PREDICTION_MODULES_CONFIGS

    node_type = "prediction"
