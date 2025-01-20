# type: ignore  # noqa: PGH003

import string
from typing import Any


class SafeFormatter(string.Formatter):
    """Utility class for loading prompt templates."""

    def get_value(self, key, args, kwargs) -> Any:  # noqa: ANN001, ANN401
        """Overloaded."""
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        return super().get_value(key, args, kwargs)

    def parse(self, format_string):  # noqa: ANN001, ANN201
        """Overloaded."""
        try:
            return super().parse(format_string)
        except ValueError:
            return [(format_string, None, None, None)]


def safe_format(format_string: str, *args: tuple[str], **kwargs: dict[str, str]) -> str:
    """Format chat template."""
    formatter = SafeFormatter()
    return formatter.format(format_string, *args, **kwargs)
