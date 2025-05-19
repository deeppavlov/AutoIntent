import logging
import sqlite3
from unittest.mock import patch

import pytest

from autointent import Pipeline
from autointent.configs import DataConfig, LoggingConfig
from autointent.custom_types import NodeType
from tests.conftest import get_search_space

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterruptAfterNCallsError(Exception):
    """Exception to simulate interruption."""


def count_trials_in_database(db_path):
    """Count the number of trials in an Optuna SQLite database."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trials")
        return cursor.fetchone()[0]


def get_completed_trial_numbers(db_path):
    """Get the trial numbers that have been completed."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT number FROM trials WHERE state = 'COMPLETE'")
        return {row[0] for row in cursor.fetchall()}


def test_pipeline_with_exception_resume(dataset_no_oos, tmp_path):
    """Test that pipeline can resume after an exception and continues from where it left off."""
    project_dir = tmp_path
    search_space = get_search_space("optuna")
    first_run_trials = set()

    call_count = 0
    max_calls_before_exception = 2

    logging_config = LoggingConfig(
        project_dir=project_dir, run_name="test_pipeline_with_exception_resume", dump_modules=True, clear_ram=True
    )
    pipeline_optimizer = Pipeline.from_search_space(search_space)
    pipeline_optimizer.set_config(logging_config)
    pipeline_optimizer.set_config(DataConfig(scheme="ho", separation_ratio=None))
    original_objective = pipeline_optimizer.nodes[NodeType.scoring].objective

    def exception_raising_objective(trial, *args, **kwargs):
        nonlocal call_count
        call_count += 1

        msg = f"First run: Processing trial #{trial.number}"
        logger.info(msg)
        if call_count >= max_calls_before_exception:
            msg = f"Simulated interruption after {call_count} calls"
            raise InterruptAfterNCallsError(msg)

        return original_objective(trial, *args, **kwargs)

    # Replace the objective with our exception-raising version
    # pipeline_optimizer.nodes[NodeType.scoring].objective = exception_raising_objective
    with (
        patch.object(pipeline_optimizer.nodes[NodeType.scoring], "objective", side_effect=exception_raising_objective),
        pytest.raises(InterruptAfterNCallsError),
    ):
        pipeline_optimizer.fit(dataset_no_oos, refit_after=False, sampler="random")

    # Verify that some trials were completed in the first run
    assert call_count > 0, "No trials were completed in the first run"

    optuna_storage_dir = logging_config.dump_dir / "optuna_storage"
    assert optuna_storage_dir.exists(), "Optuna storage directory not created in first run"

    db_files_first_run = list(optuna_storage_dir.glob("*.db"))
    trials_after_first_run = {}
    for db_file in db_files_first_run:
        trials_after_first_run[db_file.name] = count_trials_in_database(db_file)

    # Second run - should continue from where it left off
    pipeline_optimizer = Pipeline.from_search_space(search_space)
    pipeline_optimizer.set_config(logging_config)
    pipeline_optimizer.set_config(DataConfig(scheme="ho", separation_ratio=None))

    # Add tracking for second run to see which trials are executed
    second_run_trials = set()
    original_objective2 = pipeline_optimizer.nodes[NodeType.scoring].objective

    def tracking_objective2(trial, *args, **kwargs):
        msg = f"Second run: Processing trial #{trial.number}"
        logger.info(msg)
        second_run_trials.add(trial.number)
        return original_objective2(trial, *args, **kwargs)

    pipeline_optimizer.set_config(logging_config)
    pipeline_optimizer.set_config(DataConfig(scheme="ho", separation_ratio=None))
    with patch.object(pipeline_optimizer.nodes[NodeType.scoring], "objective", side_effect=tracking_objective2):
        # This run should complete without exceptions
        pipeline_optimizer.fit(dataset_no_oos, refit_after=False, sampler="random")

    # Verify the Optuna storage exists and has more trials
    db_files_second_run = list(optuna_storage_dir.glob("*.db"))

    # Key verification: The second run should NOT re-run trials from the first run
    # If they overlap completely, that means the second run restarted from scratch
    assert first_run_trials.isdisjoint(second_run_trials), (
        "Second run repeated trials from first run, which suggests it didn't resume properly: "
        f"Overlap: {first_run_trials.intersection(second_run_trials)}"
    )

    # Additionally, verify that total trials increased
    for db_file in db_files_second_run:
        if db_file.name in ["NodeType.scoring_knn.db"]:
            trials_after_second_run = count_trials_in_database(db_file)
            assert trials_after_second_run > trials_after_first_run[db_file.name], (
                f"Database {db_file.name} did not have more trials after second run "
                f"({trials_after_first_run[db_file.name]} vs {trials_after_second_run})"
            )


def test_resuming_with_memory_storage_warning(dataset_no_oos, tmp_path, caplog):
    """Test that a warning is issued when trying to resume with memory storage."""
    project_dir = tmp_path
    search_space = get_search_space("optuna")

    # Setup first pipeline with memory storage
    logging_config = LoggingConfig(
        project_dir=project_dir,
        run_name="test_memory_storage_warning",
        dump_modules=False,  # Keep modules in RAM
        clear_ram=False,
    )

    pipeline_optimizer = Pipeline.from_search_space(search_space)
    pipeline_optimizer.set_config(logging_config)
    pipeline_optimizer.set_config(DataConfig(scheme="ho", separation_ratio=None, n_folds=2, validation_size=0.2))
    pipeline_optimizer.fit(dataset_no_oos, refit_after=False, sampler="random")

    assert any(
        "Memory storage is not compatible with resuming optimization" in record.message for record in caplog.records
    )
