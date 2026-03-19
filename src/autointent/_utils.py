"""Utils."""

import importlib
from typing import TypeVar

import torch

T = TypeVar("T")


def _funcs_to_dict(*funcs: T) -> dict[str, T]:
    """Convert functions to a dictionary.

    Args:
        *funcs: Functions to convert
    Returns:
        Dictionary of functions
    """
    return {func.__name__: func for func in funcs}  # type: ignore[attr-defined]


def detect_device() -> str:
    """Automatically detects CUDA, MPS and CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


def require(dependency: str, extra: str | None = None) -> None:
    """Try to import dependency, raise informative ImportError if missing.

    Args:
        dependency: The name of the module to import
        extra: Optional extra package name for pip install instructions

    Returns:
        The imported module

    Raises:
        ImportError: If the dependency is not installed
    """
    try:
        importlib.import_module(dependency)
    except ImportError as e:
        extra_info = f" Install with `pip install autointent[{extra}]`." if extra else ""
        msg = f"Missing dependency '{dependency}' required for this feature.{extra_info}"
        raise ImportError(msg) from e
