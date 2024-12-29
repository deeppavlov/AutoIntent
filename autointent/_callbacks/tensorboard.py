from pathlib import Path
from typing import Any

from autointent._callbacks.base import OptimizerCallback


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

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Log final metrics.

        :param metrics: Final metrics.
        """
        if self.module_writer is None:
            msg = "start_run must be called before log_final_metrics."
            raise RuntimeError(msg)

        log_dir = Path(self.dirpath) / "final_metrics"
        self.module_writer = self.writer(log_dir=log_dir)  # type: ignore[no-untyped-call]

        for key, value in metrics.items():
            if isinstance(value, int | float):
                self.module_writer.add_scalar(key, value)  # type: ignore[no-untyped-call]
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
