import platform

import pytest


@pytest.fixture
def on_windows() -> bool:
    return platform.system() == "Windows"
