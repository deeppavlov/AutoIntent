"""Session-scoped OpenSearch testcontainer."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session")
def opensearch_container() -> Iterator[tuple[str, int]]:
    """Boot an OpenSearch container for the test session; yield (host, port)."""
    from testcontainers.opensearch import OpenSearchContainer

    with OpenSearchContainer() as container:
        yield container.get_config()["host"], int(container.get_config()["port"])
