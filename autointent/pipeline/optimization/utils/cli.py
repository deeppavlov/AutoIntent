import importlib.resources as ires
from logging import Logger
from pathlib import Path
from typing import Any

import yaml


def get_logs_dir(run_name: str, logs_dir: Path | None = None) -> Path:
    if logs_dir is None:
        logs_dir = Path.cwd()
    res = logs_dir / run_name
    res.mkdir(parents=True)
    return res


def load_config(config_path: str | Path | None, multilabel: bool, logger: Logger | None = None) -> dict[str, Any]:
    """load config from the given path or load default config which is distributed along with the autointent package"""
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
