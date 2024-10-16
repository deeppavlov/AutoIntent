from collections.abc import Mapping
from typing import ClassVar

from autointent.configs.modules import ModuleConfig
from autointent.metrics import METRIC_FN
from autointent.modules import Module


class NodeInfo:
    metrics_available: ClassVar[Mapping[str, METRIC_FN]]
    modules_available: ClassVar[Mapping[str, type[Module]]]
    modules_configs: ClassVar[Mapping[str, type[ModuleConfig]]]
    node_type: str
