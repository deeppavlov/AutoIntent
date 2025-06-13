from typing import Literal

from autointent._callbacks.base import OptimizerCallback
from autointent._callbacks.callback_handler import CallbackHandler
from autointent._callbacks.emissions_tracker import EmissionsTrackerCallback
from autointent._callbacks.tensorboard import TensorBoardCallback
from autointent._callbacks.wandb import WandbCallback

REPORTERS = {cb.name: cb for cb in [WandbCallback, TensorBoardCallback, EmissionsTrackerCallback]}

REPORTERS_NAMES = Literal[tuple(REPORTERS.keys())]  # type: ignore[valid-type]


def get_callbacks(reporters: list[str] | None) -> CallbackHandler:
    """Get the list of callbacks.

    Args:
        reporters: List of reporters to use.

    Returns:
        CallbackHandler: Callback handler.
    """
    if not reporters:
        return CallbackHandler()

    reporters_cb = []
    for reporter in reporters:
        if reporter not in REPORTERS:
            msg = f"Reporter {reporter} not supported. Supported reporters {','.join(REPORTERS)}"
            raise ValueError(msg)
        reporters_cb.append(REPORTERS[reporter])
    return CallbackHandler(callbacks=reporters_cb)


__all__ = [
    "REPORTERS_NAMES",
    "CallbackHandler",
    "EmissionsTrackerCallback",
    "OptimizerCallback",
    "TensorBoardCallback",
    "WandbCallback",
    "get_callbacks",
]
