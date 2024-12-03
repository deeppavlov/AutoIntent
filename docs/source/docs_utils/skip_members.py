from typing import Any

from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options


def skip_member(app: Sphinx, what: str, name: str, obj: Any, skip: bool, options: Options) -> bool | None:  # noqa: ANN401, ARG001
    """
    Custom function to skip members based on docstring tags.

    Args:
        app (Sphinx): The Sphinx application instance.
        what (str): The type of the object (e.g., 'module', 'class', 'method').
        name (str): The name of the object.
        obj (Any): The object itself.
        skip (bool): Whether to skip the object by default.
        options (Options): Options for the autodoc directive.

    Returns:
        bool | None: True to skip the member, None to use the default behavior.
    """
    # Check if the member has a docstring
    if hasattr(obj, "__doc__") and obj.__doc__ and "# exclude from docs" in obj.__doc__:
        return True
    return skip
