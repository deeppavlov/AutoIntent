from collections.abc import Callable
from typing import ClassVar

from autointent.modules import Module


class NodeInfo:
    metrics_available: ClassVar[dict[str, Callable]]
    modules_available: ClassVar[dict[str, type[Module]]]
    node_type: str
