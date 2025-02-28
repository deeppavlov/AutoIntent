"""Utils."""

from typing import TypeVar

T = TypeVar("T")


def _funcs_to_dict(*funcs: T) -> dict[str, T]:
    """Convert functions to a dictionary.

    Args:
        *funcs: Functions to convert
    Returns:
        Dictionary of functions
    """
    return {func.__name__: func for func in funcs}  # type: ignore[attr-defined]
