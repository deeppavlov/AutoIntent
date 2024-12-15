"""Base class for reporters (W&B, TensorBoard, etc)."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class OptimizerCallback(ABC):
    """Base class for reporters (W&B, TensorBoard, etc)."""

    # Implementation inspired by TrainerCallback from HuggingFace Transformers. https://github.com/huggingface/transformers/blob/91b8ab18b778ae9e2f8191866e018cd1dc7097be/src/transformers/trainer_callback.py#L260
    name: str

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def start_run(self, run_name: str, dirpath: Path) -> None:
        """
        Start a new run.

        :param run_name: Name of the run.
        :param dirpath: Path to the directory where the logs will be saved.
        """

    @abstractmethod
    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        """
        Start a new module.

        :param module_name: Name of the module.
        :param num: Number of the module.
        :param module_kwargs: Module parameters.
        """

    @abstractmethod
    def log_value(self, **kwargs: dict[str, Any]) -> None:
        """
        Log data.

        :param kwargs: Data to log.
        """

    @abstractmethod
    def end_module(self) -> None:
        """End a module."""

    @abstractmethod
    def end_run(self) -> None:
        """End a run."""


class CallbackHandler(OptimizerCallback):
    """Internal class that just calls the list of callbacks in order."""

    callbacks: list[OptimizerCallback]

    def __init__(self, callbacks: list[type[OptimizerCallback]] | None = None) -> None:
        """Initialize the callback handler."""
        if not callbacks:
            self.callbacks = []
            return

        self.callbacks = [cb() for cb in callbacks]

    def start_run(self, run_name: str, dirpath: Path) -> None:
        """
        Start a new run.

        :param run_name: Name of the run.
        :param dirpath: Path to the directory where the logs will be saved.
        """
        self.call_events("start_run", run_name=run_name, dirpath=dirpath)

    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        """
        Start a new module.

        :param module_name: Name of the module.
        :param num: Number of the module.
        :param module_kwargs: Module parameters.
        """
        self.call_events("start_module", module_name=module_name, num=num, module_kwargs=module_kwargs)

    def log_value(self, **kwargs: dict[str, Any]) -> None:
        """
        Log data.

        :param kwargs: Data to log.
        """
        self.call_events("log_value", **kwargs)

    def end_module(self) -> None:
        """End a module."""
        self.call_events("end_module")

    def end_run(self) -> None:
        self.call_events("end_run")

    def call_events(self, event: str, **kwargs: Any) -> None:  # noqa: ANN401
        for callback in self.callbacks:
            getattr(callback, event)(**kwargs)


class WandbCallback(OptimizerCallback):
    """
    Wandb callback.

    This callback logs the optimization process to W&B.
    To specify the project name, set the `WANDB_PROJECT` environment variable. Default is `autointent`.
    """

    name = "wandb"

    def __init__(self) -> None:
        """Initialize the callback."""
        try:
            import wandb
        except ImportError:
            msg = "Please install wandb to use this callback. `pip install wandb`"
            raise ImportError(msg) from None

        self.wandb = wandb

    def start_run(self, run_name: str, dirpath: Path) -> None:
        """
        Start a new run.

        :param run_name: Name of the run.
        :param dirpath: Path to the directory where the logs will be saved. (Not used for this callback)
        """
        self.project_name = os.getenv("WANDB_PROJECT", "autointent")
        self.group = run_name
        self.dirpath = dirpath

    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        """
        Start a new module.

        :param module_name: Name of the module.
        :param num: Number of the module.
        :param module_kwargs: Module parameters.
        """
        self.wandb.init(
            project=self.project_name,
            group=self.group,
            name=f"{module_name}_{num}",
            config=module_kwargs,
        )

    def log_value(self, **kwargs: dict[str, Any]) -> None:
        """
        Log data.

        :param kwargs: Data to log.
        """
        self.wandb.log(kwargs)

    def end_module(self) -> None:
        """End a module."""
        self.wandb.finish()

    def end_run(self) -> None:
        pass


class TensorBoardCallback(OptimizerCallback):
    """
    TensorBoard callback.

    This callback logs the optimization process to TensorBoard.
    """

    name = "tensorboard"

    def __init__(self) -> None:
        """Initialize the callback."""
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]

            self.writer = SummaryWriter
        except ImportError:
            try:
                from tensorboardX import SummaryWriter  # type: ignore[no-redef]

                self.writer = SummaryWriter
            except ImportError:
                msg = (
                    "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                    " install tensorboardX."
                )
                raise ImportError(msg) from None

    def start_run(self, run_name: str, dirpath: Path) -> None:
        """
        Start a new run.

        :param run_name: Name of the run.
        :param dirpath: Path to the directory where the logs will be saved.
        """
        self.run_name = run_name
        self.dirpath = dirpath

    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        """
        Start a new module.

        :param module_name: Name of the module.
        :param num: Number of the module.
        :param module_kwargs: Module parameters.
        """
        module_run_name = f"{self.run_name}_{module_name}_{num}"
        log_dir = Path(self.dirpath) / module_run_name
        self.module_writer = self.writer(log_dir=log_dir)  # type: ignore[no-untyped-call]

        self.module_writer.add_text("module_info", f"Starting module {module_name}_{num}")  # type: ignore[no-untyped-call]
        for key, value in module_kwargs.items():
            self.module_writer.add_text(f"module_params/{key}", str(value))  # type: ignore[no-untyped-call]

    def log_value(self, **kwargs: dict[str, Any]) -> None:
        """
        Log data.

        :param kwargs: Data to log.
        """
        if self.module_writer is None:
            msg = "start_run must be called before log_value."
            raise RuntimeError(msg)

        for key, value in kwargs.items():
            if isinstance(value, int | float):
                self.module_writer.add_scalar(key, value)
            else:
                self.module_writer.add_text(key, str(value))  # type: ignore[no-untyped-call]

    def end_module(self) -> None:
        """End a module."""
        if self.module_writer is None:
            msg = "start_run must be called before end_module."
            raise RuntimeError(msg)

        self.module_writer.add_text("module_info", "Ending module")  # type: ignore[no-untyped-call]
        self.module_writer.close()  # type: ignore[no-untyped-call]

    def end_run(self) -> None:
        pass


REPORTERS = {cb.name: cb for cb in [WandbCallback, TensorBoardCallback]}


def get_callbacks(reporters: list[str] | None) -> CallbackHandler:
    """
    Get the list of callbacks.

    :param reporters: List of reporters to use.
    :return: Callback handler.
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
