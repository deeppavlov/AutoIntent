"""Utils."""

import importlib
from typing import Any, TypeVar

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


def _is_package_available(pkg_name: str) -> bool:
    return importlib.util.find_spec(pkg_name) is not None


def _requires_package(
    obj: Any,  # noqa: ANN401
    package_name: str,
    group_name: str,
) -> None:
    """Check if a package is available and raise an error with installation instructions if it's not.

    Args:
        obj: The object (class or function) that requires the package.
        package_name: The name of the package to check.
        group_name: The instruction to install the package. If None, defaults to "pip install {package_name}".
    """
    if _is_package_available(package_name):
        return
    install_instruction = f"pip install {package_name}"
    group_install_instruction = f"pip install autointent[{group_name}]"
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    msg = (
        f"{name} requires the `{package_name}` library but it was not found in your environment. "
        f"Please run `{group_install_instruction}` or `{install_instruction}` to install the package."
    )
    raise ImportError(msg)
