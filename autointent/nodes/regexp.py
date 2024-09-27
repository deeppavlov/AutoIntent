from collections.abc import Callable
from typing import ClassVar

from autointent.metrics import regexp_partial_accuracy, regexp_partial_precision
from autointent.modules import RegExp

from .base import Node


class RegExpNode(Node):
    metrics_available: ClassVar[dict[str, Callable]] = {
        "regexp_partial_precision": regexp_partial_precision,
        "regexp_partial_accuracy": regexp_partial_accuracy,
    }

    modules_available: ClassVar[dict[str, Callable]] = {"regexp": RegExp}

    node_type = "regexp"
