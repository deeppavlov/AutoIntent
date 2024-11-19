"""Utility functions for the vector index client."""

from pathlib import Path
from uuid import uuid4


def get_db_dir(db_dir: str | Path | None = None) -> Path:
    """
    Get the directory for the vector database.

    :param db_dir: Directory for the vector database.
    :return: Path to the vector database directory.
    """
    db_dir = Path.cwd() / ("vector_db_" + str(uuid4())) if db_dir is None else Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir
