"""Cli endpoint."""
import importlib.resources as ires
import logging
from logging import Logger
from pathlib import Path
from typing import Any

import hydra
import yaml

from autointent import Context
from autointent.configs._optimization_cli import OptimizationConfig

from ._pipeline import Pipeline


@hydra.main(config_name="optimization_config", config_path=".", version_base=None)
def optimize(cfg: OptimizationConfig) -> None:
    """
    Run the optimization pipeline.

    :param cfg: Configuration for the optimization pipeline
    :return:
    """
    logger = logging.getLogger(__name__)

    logger.debug("Run Name: %s", cfg.logs.run_name)
    logger.debug("logs and assets: %s", cfg.logs.dirpath)
    logger.debug("Vector index path: %s", cfg.vector_index.db_dir)

    # create shared objects for a whole pipeline
    context = Context(cfg.seed)
    context.configure_logging(cfg.logs)
    context.configure_vector_index(cfg.vector_index, cfg.embedder)
    context.configure_data(cfg.data)

    # run optimization
    search_space_config = load_config(cfg.task.search_space_path, context.is_multilabel(), logger)
    pipeline = Pipeline.from_search_space(search_space_config)
    pipeline._fit(context)  # noqa: SLF001

    # save results
    context.dump()


def load_config(config_path: str | Path | None, multilabel: bool, logger: Logger | None = None) -> dict[str, Any]:
    """
    Load configuration from the given path or load default configuration.

    :param config_path: Path to the configuration file
    :param multilabel: Whether to use multilabel or not
    :param logger: Logger
    :return:
    """
    if config_path is not None:
        if logger is not None:
            logger.debug("loading optimization search space config from %s...)", config_path)
        with Path(config_path).open() as file:
            file_content = file.read()
    else:
        if logger is not None:
            logger.debug("loading default optimization search space config...")
        config_name = "default-multilabel-config.yaml" if multilabel else "default-multiclass-config.yaml"
        with ires.files("autointent.datafiles").joinpath(config_name).open() as file:
            file_content = file.read()
    return yaml.safe_load(file_content)  # type: ignore[no-any-return]
