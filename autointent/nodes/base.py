from collections.abc import Callable
from typing import ClassVar

from autointent.modules import Module


class NodeInfo:
    metrics_available: ClassVar[dict[str, Callable]]
    modules_available: ClassVar[dict[str, type[Module]]]
    modules_configs: ClassVar[dict[str, type]]
    node_type: str
