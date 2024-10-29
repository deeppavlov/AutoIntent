import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from appdirs import user_cache_dir, user_config_dir


def get_logger() -> logging.Logger:
    logger = logging.getLogger("my_logger")

    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger


@dataclass
class ChromaConfig:
    cache_directories: list[str] = field(default_factory=list)


def get_chroma_cache_dir() -> Path:
    """Get system's default cache dir."""
    cache_dir = user_cache_dir("autointent")
    return Path(cache_dir) / "chroma"


def get_chroma_config_path() -> Path:
    """Get system's default config dir."""
    config_dir = user_config_dir("autointent")
    return Path(config_dir) / "chromadb.json"


def read_chroma_config() -> ChromaConfig:
    path = get_chroma_config_path()
    if not path.exists():
        return ChromaConfig()
    with path.open() as file:
        return ChromaConfig(**json.load(file))


def write_chroma_config(config: ChromaConfig) -> None:
    path = get_chroma_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(asdict(config), file, ensure_ascii=False, indent=4)


def add_cache_directory(directory: str) -> None:
    """Save path into chroma config in order to remove it from cache later."""
    chroma_config = read_chroma_config()

    directories = set(chroma_config.cache_directories)
    directories.add(directory)
    chroma_config.cache_directories = sorted(directories)

    write_chroma_config(chroma_config)


def get_db_dir(db_dir: str | Path | None = None) -> Path:
    """
    Get the directory path for chroma db file.
    Use default cache dir if not provided.
    Save path into user config in order to remove it from cache later.
    """

    db_dir = get_chroma_cache_dir() / str(uuid4()) if db_dir is None else Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    add_cache_directory(str(db_dir.resolve()))

    return db_dir


def clear_chroma_cache() -> None:
    # TODO: test on all platforms
    logger = get_logger()
    chroma_config = read_chroma_config()
    for cache_dirs in chroma_config.cache_directories:
        if Path(cache_dirs).exists():
            shutil.rmtree(cache_dirs)
            logger.info("cleared vector index at %s", cache_dirs)
        else:
            logger.error("vector index at %s not found", cache_dirs)
        chroma_config.cache_directories.remove(cache_dirs)
    write_chroma_config(chroma_config)


def clear_specific_cache(directory: str) -> None:
    """TODO test this code"""
    chroma_config = read_chroma_config()
    if directory in chroma_config.cache_directories:
        try:
            shutil.rmtree(directory)
            chroma_config.cache_directories.remove(directory)
            write_chroma_config(chroma_config)
        except OSError:
            pass
    else:
        pass