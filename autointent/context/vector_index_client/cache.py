from pathlib import Path
from uuid import uuid4


def get_db_dir(db_dir: str | Path | None = None) -> Path:
    """
    Get the directory path for chroma db file.
    Use default cache dir if not provided.
    Save path into user config in order to remove it from cache later.
    """

    root = Path(db_dir) if db_dir is not None else Path.cwd()
    db_dir = root / "vector_db" / str(uuid4()) if db_dir is None else Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    return db_dir
