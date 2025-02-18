"""Configuration for the optimization process."""

from pathlib import Path

from pydantic import BaseModel, Field, PositiveInt, field_validator

from autointent._callbacks import REPORTERS_NAMES
from autointent.custom_types import FloatFromZeroToOne, SamplerType, ValidationScheme

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


class TaskConfig(BaseModel):
    """Configuration for the task to optimize."""

    search_space_path: Path | None = None
    """Path to the search space configuration file. If None, the default search space will be used"""
    sampler: SamplerType = "brute"


class LoggingConfig(BaseModel):
    """Configuration for the logging."""

    project_dir: Path = Field(default_factory=lambda: Path.cwd() / "runs")
    """Path to the directory with different runs."""
    run_name: str = Field(default_factory=get_run_name)
    """Name of the run. If None, a random name will be generated"""
    dump_modules: bool = False
    """Whether to dump the modules or not"""
    clear_ram: bool = False
    """Whether to clear the RAM after dumping the modules"""
    report_to: list[str] | None = None
    """List of callbacks to report to. If None, no callbacks will be used"""

    @property
    def dirpath(self) -> Path:
        """Path to the directory where the logs will be saved."""
        if not hasattr(self, "_dirpath"):
            self._dirpath = self.project_dir / self.run_name
        return self._dirpath

    @property
    def dump_dir(self) -> Path:
        """Path to the directory where the modules will be dumped."""
        if not hasattr(self, "_dump_dir"):
            self._dump_dir = self.dirpath / "modules_dumps"
        return self._dump_dir

    @field_validator("report_to")
    @classmethod
    def validate_report_to(cls, v: list[str] | None) -> list[str] | None:
        """Validate the report_to field."""
        if v is None:
            return None
        for reporter in v:
            if reporter not in REPORTERS_NAMES:
                msg = f"Reporter {reporter} is not supported. Supported reporters: {REPORTERS_NAMES}"
                raise ValueError(msg)
        return v


class VectorIndexConfig(BaseModel):
    """Configuration for the vector index."""

    save_db: bool = False
    """Whether to save the vector index database or not"""
