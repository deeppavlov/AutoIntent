"""Separate file to fix circular import error."""

from pathlib import Path
from typing import Any

from autointent.generation import Generator

from .base import BaseObjectDumper


class GeneratorDumper(BaseObjectDumper[Generator]):
    dir_or_file_name = "generators"

    @staticmethod
    def dump(obj: Generator, path: Path, exists_ok: bool) -> None:
        obj.dump(path, exist_ok=exists_ok)

    @staticmethod
    def load(path: Path, **kwargs: Any) -> Generator:  # noqa: ANN401, ARG004
        return Generator.load(path)

    @classmethod
    def check_isinstance(cls, obj: Any) -> bool:  # noqa: ANN401
        return isinstance(obj, Generator)
