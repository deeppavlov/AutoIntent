import json
import os
import shutil
from dataclasses import asdict, dataclass, field

from appdirs import user_cache_dir, user_config_dir


@dataclass
class ChromaConfig:
    cache_directories: list[str] = field(default_factory=list)


def get_chroma_cache_dir():
    """Get system's default cache dir."""
    cache_dir = user_cache_dir("autointent")
    res = os.path.join(cache_dir, "chroma")
    return res


def get_chroma_config_path():
    """Get system's default config dir."""
    config_dir = user_config_dir("autointent")
    res = os.path.join(config_dir, "chromadb.json")
    return res


def read_chroma_config():
    path = get_chroma_config_path()
    if not os.path.exists(path):
        return ChromaConfig()
    with open(path) as file:
        config = ChromaConfig(**json.load(file))
    return config


def write_chroma_config(config: ChromaConfig):
    path = get_chroma_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(asdict(config), file, ensure_ascii=False, indent=4)


def add_cache_directory(directory):
    """Save path into chroma config in order to remove it from cache later."""
    chroma_config = read_chroma_config()

    directories = set(chroma_config.cache_directories)
    directories.add(directory)
    chroma_config.cache_directories = sorted(directories)

    write_chroma_config(chroma_config)


def get_db_dir(db_dir: os.PathLike, run_name: str):
    """
    Get the directory path for chroma db file.
    Use default cache dir if not provided.
    Save path into user config in order to remove it from cache later.
    """

    if db_dir == "":
        cache_dir = get_chroma_cache_dir()
        db_dir = os.path.join(cache_dir, run_name)

    add_cache_directory(db_dir)

    return db_dir


def clear_chroma_cache():
    # TODO: test on all platforms
    chroma_config = read_chroma_config()
    for cache_dirs in chroma_config.cache_directories:
        if os.path.exists(cache_dirs):
            shutil.rmtree(cache_dirs)
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
        except OSError as e:
            print(f"Error removing cache directory {directory}: {e}")
    else:
        print(f"Directory {directory} not found in cache")
