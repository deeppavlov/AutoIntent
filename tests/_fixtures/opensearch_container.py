"""Session-scoped OpenSearch testcontainer."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def opensearch_container():
    """Boot an OpenSearch container for the test session; yield (host, port)."""
    from testcontainers.opensearch import OpenSearchContainer

    with OpenSearchContainer() as container:
        yield container.get_config()["host"], int(container.get_config()["port"])
