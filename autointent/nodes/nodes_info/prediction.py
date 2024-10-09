from collections.abc import Callable
from typing import ClassVar

from autointent.configs.modules import PREDICTION_MODULES_CONFIGS
from autointent.metrics import PREDICTION_METRICS_MULTICLASS, PREDICTION_METRICS_MULTILABEL
from autointent.modules import PREDICTION_MODULES_MULTICLASS, PREDICTION_MODULES_MULTILABEL, Module

from .base import NodeInfo


class PredictionNodeInfo(NodeInfo):
    metrics_available: ClassVar[dict[str, Callable]] = PREDICTION_METRICS_MULTICLASS | PREDICTION_METRICS_MULTILABEL

    modules_available: ClassVar[dict[str, type[Module]]] = PREDICTION_MODULES_MULTICLASS | PREDICTION_MODULES_MULTILABEL

    modules_configs: ClassVar[dict[str, type]] = PREDICTION_MODULES_CONFIGS

    node_type = "prediction"
