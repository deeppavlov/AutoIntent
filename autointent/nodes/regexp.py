from ..metrics import regexp_partial_precision, regexp_partial_accuracy
from ..modules import RegExp
from .base import Node


class RegExpNode(Node):
    metrics_available = {
        "regexp_partial_precision": regexp_partial_precision,
        "regexp_partial_accuracy": regexp_partial_accuracy,
    }

    modules_available = {"regexp": RegExp}

    node_type = "regexp"
