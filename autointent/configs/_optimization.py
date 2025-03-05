"""Configuration for the optimization process."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from autointent._callbacks import REPORTERS_NAMES
from autointent.custom_types import FloatFromZeroToOne, ValidationScheme

from ._name import get_run_name


class DataConfig(BaseModel):
    """Configuration for the data used in the optimization process."""

    model_config = ConfigDict(extra="forbid")
    scheme: ValidationScheme = Field("ho", description="Validation scheme to use.")
    """Hold-out or cross-validation."""
    n_folds: PositiveInt = Field(3, description="Number of folds in cross-validation.")
    """Number of folds in cross-validation."""
    validation_size: FloatFromZeroToOne = Field(
        0.2,
        description=(
            "Fraction of train samples to allocate for validation (if input dataset doesn't contain validation split)."
        ),
    )
    """Fraction of train samples to allocate for validation (if input dataset doesn't contain validation split)."""
    separation_ratio: FloatFromZeroToOne | None = Field(
        0.5, description="Set to float to prevent data leak between scoring and decision nodes."
    )
    """Set to float to prevent data leak between scoring and decision nodes."""


class LoggingConfig(BaseModel):
    """Configuration for the logging."""

    model_config = ConfigDict(extra="forbid")

    _dirpath: Path | None = None
    _dump_dir: Path | None = None

    project_dir: Path | str | None = Field(None, description="Path to the directory with different runs.")
    """Path to the directory with different runs."""
    run_name: str | None = Field(None, description="Name of the run. If None, a random name will be generated.")
    """Name of the run. If None, a random name will be generated.
    To get run_name better use :py:meth:`autointent.configs.LoggingConfig.get_run_name`."""
    dump_modules: bool = Field(False, description="Whether to dump the modules or not")
    """Whether to dump the modules or not"""
    clear_ram: bool = Field(False, description="Whether to clear the RAM after dumping the modules")
    """Whether to clear the RAM after dumping the modules"""
    report_to: list[REPORTERS_NAMES] | None = Field(  # type: ignore[valid-type]
        None, description="List of callbacks to report to. If None, no callbacks will be used"
    )
    """List of callbacks to report to. If None, no callbacks will be used"""

    @property
    def dirpath(self) -> Path:
        """Path to the directory where the logs will be saved."""
        if self._dirpath is None:
            project_dir = Path.cwd() / "runs" if self.project_dir is None else Path(self.project_dir)
            self._dirpath = project_dir / self.get_run_name()
        return self._dirpath

    @property
    def dump_dir(self) -> Path:
        """Path to the directory where the modules will be dumped."""
        if self._dump_dir is None:
            self._dump_dir = self.dirpath / "modules_dumps"
        return self._dump_dir

    def get_run_name(self) -> str:
        """Return name of the run.

        Use this method instead of direct adressing to :py:attr:`autointent.configs.LoggingConfig.run_name`.
        """
        if self.run_name is None:
            self.run_name = get_run_name()
        return self.run_name
