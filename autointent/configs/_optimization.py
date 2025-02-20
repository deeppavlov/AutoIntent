"""Configuration for the optimization process."""

from pathlib import Path

from pydantic import BaseModel, Field, PositiveInt

from autointent._callbacks import REPORTERS_NAMES
from autointent.custom_types import FloatFromZeroToOne, ValidationScheme

from ._name import get_run_name


class DataConfig(BaseModel):
    """Configuration for the data used in the optimization process."""

    scheme: ValidationScheme = "ho"
    """Hold-out or cross-validation."""
    n_folds: PositiveInt = 3
    """Number of folds in cross-validation."""
    validation_size: FloatFromZeroToOne = 0.2
    """Fraction of train samples to allocate for validation (if input dataset doesn't contain validation split)."""
    separation_ratio: FloatFromZeroToOne | None = 0.5
    """Set to float to prevent data leak between scoring and decision nodes."""


class LoggingConfig(BaseModel):
    """Configuration for the logging."""

    project_dir: Path | str = Field(default_factory=lambda: Path.cwd() / "runs")
    """Path to the directory with different runs."""
    run_name: str = Field(default_factory=get_run_name)
    """Name of the run. If None, a random name will be generated"""
    dump_modules: bool = False
    """Whether to dump the modules or not"""
    clear_ram: bool = False
    """Whether to clear the RAM after dumping the modules"""
    report_to: list[REPORTERS_NAMES] | None = None  # type: ignore[valid-type]
    """List of callbacks to report to. If None, no callbacks will be used"""

    @property
    def dirpath(self) -> Path:
        """Path to the directory where the logs will be saved."""
        if not hasattr(self, "_dirpath"):
            self._dirpath = Path(self.project_dir) / self.run_name
        return self._dirpath

    @property
    def dump_dir(self) -> Path:
        """Path to the directory where the modules will be dumped."""
        if not hasattr(self, "_dump_dir"):
            self._dump_dir = self.dirpath / "modules_dumps"
        return self._dump_dir


class VectorIndexConfig(BaseModel):
    """Configuration for the vector index."""

    save_db: bool = False
    """Whether to save the vector index database or not"""
