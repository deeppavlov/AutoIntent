"""Configuration for the optimization process."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from autointent._callbacks import REPORTERS_NAMES
from autointent.custom_types import FloatFromZeroToOne, SamplerType, ValidationScheme

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
            "Fraction of train samples to allocate for validation (if input dataset doesn't contain validation split). "
            "If `is_few_shot_train` is True, this value will be ignored."
        ),
    )
    """Fraction of train samples to allocate for validation (if input dataset doesn't contain validation split)."""
    separation_ratio: FloatFromZeroToOne | None = Field(
        0.5, description="Set to float to prevent data leak between scoring and decision nodes."
    )
    """Set to float to prevent data leak between scoring and decision nodes."""
    is_few_shot_train: bool = Field(False, description="Whether to use few-shot training.")
    """Whether to use few-shot training."""
    examples_per_intent: PositiveInt = Field(
        8,
        description="Number of examples per intent for few-shot validation. If None, all examples will be used.",
    )
    """Number of examples per intent for few-shot validation. If None, all examples will be used."""


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
    log_interval_time: float = Field(
        0.1, description="Sampling interval for the system monitor in seconds for Wandb logger."
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


class HPOConfig(BaseModel):
    """Configuration for hyperparameter optimization using Optuna.

    For more detailed information about the TPE sampler and its parameters,
    refer to Optuna's documentation of `TPESampler
    <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html>`_,
    `study.optimize
    <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize>`_,
    `RandomSampler <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html>`_.
    """

    model_config = ConfigDict(extra="forbid")

    # optuna generic
    sampler: SamplerType = Field(default="tpe", description="Optuna sampler. One of 'tpe', 'random'.")
    n_trials: int = Field(
        15,  # small value for tests
        description=(
            "Number of trials to run in the optimization process. "
            "This is the total number of different hyperparameter combinations that will be evaluated."
        ),
    )
    timeout: float | None = Field(
        None,
        description=(
            "Time limit in seconds for the optimization process. "
            "If None, the optimization will run until n_trials is reached."
        ),
    )
    n_jobs: int = Field(
        1,
        description="Number of parallel jobs to run. Set to -1 to use all available CPU cores.",
    )

    # tpe sampler specific
    n_startup_trials: int = Field(
        10,  # small value for tests
        description=(
            "Number of initial trials to run using random sampling before switching to TPE algorithm. "
            "This helps in better initialization of the TPE algorithm."
        ),
    )
    consider_prior: bool = Field(
        True,
        description=(
            "Whether to use Gaussian (normal) distribution as prior for integer and float parameter spaces. "
            "This helps in better initialization of the TPE algorithm's parameter distributions."
        ),
    )
    prior_weight: int = Field(
        1,
        description=(
            "Weight of the prior distribution in the TPE algorithm. "
            "Higher values make the algorithm more conservative in exploring new regions."
        ),
    )
    n_ei_candidates: int = Field(
        24,
        description=(
            "Number of candidates to sample for expected improvement calculation in TPE algorithm. "
            "Higher values may lead to better exploration but slower optimization."
        ),
    )
    constant_liar: bool = Field(
        False,
        description=(
            "Whether to use constant liar strategy for parallel optimization. "
            "If True, the algorithm will penalize running trials to avoid suggesting parameter configurations "
            "that are too close to currently running trials."
        ),
    )
