import os
from pathlib import Path

from appdirs import user_cache_dir


def get_cache_dir() -> Path:
    """Get system's default cache dir."""
    cache_dir = user_cache_dir("autointent")
    return Path(cache_dir) / "chroma"


def get_db_dir(run_name: str, db_dir: str | None = None) -> str:
    """
    Get the directory path for chroma db file.
    Use default cache dir if not provided.
    Save path into user config in order to remove it from cache later.
    """

    if db_dir is None:
        cache_dir = get_cache_dir()
        db_dir = os.path.join(cache_dir, run_name)  # noqa: PTH118
    return db_dir
