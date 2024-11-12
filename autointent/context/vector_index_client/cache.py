from pathlib import Path
from uuid import uuid4


def get_db_dir(db_dir: str | Path | None = None) -> Path:
    db_dir = Path.cwd() / "vector_db" / str(uuid4()) if db_dir is None else Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir
