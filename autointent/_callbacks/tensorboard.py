from pathlib import Path
from typing import Any

from autointent._callbacks.base import OptimizerCallback


class TensorBoardCallback(OptimizerCallback):
    """TensorBoard callback for logging the optimization process."""

    name = "tensorboard"

    def __init__(self) -> None:
        """Initializes the TensorBoard callback.

        Attempts to import `torch.utils.tensorboard` first. If unavailable, tries to import `tensorboardX`.
        Raises an ImportError if neither are installed.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

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

    def start_run(self, run_name: str, dirpath: Path, log_interval_time: float) -> None:
        """Starts a new run and sets the directory for storing logs.

        Args:
            run_name: Name of the run.
            dirpath: Path to the directory where logs will be saved.
            log_interval_time: Sampling interval for the system monitor in seconds.
        """
        self.run_name = run_name
        self.dirpath = dirpath
        self.log_interval_time = log_interval_time

    def start_module(self, module_name: str, num: int, module_kwargs: dict[str, Any]) -> None:
        """Starts a new module and initializes a TensorBoard writer for it.

        Args:
            module_name: Name of the module.
            num: Identifier number of the module.
            module_kwargs: Dictionary containing module parameters.
        """
        module_run_name = f"{self.run_name}_{module_name}_{num}"
        log_dir = Path(self.dirpath) / module_run_name
        self.module_writer = self.writer(log_dir=log_dir)  # type: ignore[no-untyped-call]

        self.module_writer.add_text("module_info", f"Starting module {module_name}_{num}")  # type: ignore[no-untyped-call]
        for key, value in module_kwargs.items():
            self.module_writer.add_text(f"module_params/{key}", str(value))  # type: ignore[no-untyped-call]

    def log_value(self, **kwargs: dict[str, int | float | Any]) -> None:
        """Logs scalar or text values.

        Args:
            **kwargs: Key-value pairs of data to log. Scalars will be logged as numerical values, others as text.
        """
        for key, value in kwargs.items():
            if isinstance(value, int | float):
                self.module_writer.add_scalar(key, value)
            else:
                self.module_writer.add_text(key, str(value))  # type: ignore[no-untyped-call]

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Logs training metrics.

        Args:
            metrics: Dictionary of metrics to log.
        """
        for key, value in metrics.items():
            if isinstance(value, int | float):
                self.module_writer.add_scalar(key, value)  # type: ignore[no-untyped-call]
            else:
                self.module_writer.add_text(key, str(value))  # type: ignore[no-untyped-call]

    def log_final_metrics(self, metrics: dict[str, Any]) -> None:
        """Logs final metrics at the end of training.

        Args:
            metrics: Dictionary of final metrics.

        Raises:
            RuntimeError: If `start_run` has not been called before logging final metrics.
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
        """Ends the current module and closes the TensorBoard writer.

        Raises:
            RuntimeError: If `start_run` has not been called before ending the module.
        """
        if self.module_writer is None:
            msg = "start_run must be called before end_module."
            raise RuntimeError(msg)

        self.module_writer.add_text("module_info", "Ending module")  # type: ignore[no-untyped-call]
        self.module_writer.close()  # type: ignore[no-untyped-call]

    def end_run(self) -> None:
        """Ends the current run. This method is currently a placeholder."""
