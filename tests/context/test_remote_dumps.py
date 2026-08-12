"""Tests for remote-aware dump deletion (issue #343)."""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from pathlib import Path

import pytest

from autointent import remove_module_dump

from .test_vector_index import _DOCKER_AVAILABLE, _one_hot_docs, _os_backend


def test_remove_module_dump_plain_dir(tmp_path: Path) -> None:
    """Dumps without manifests are removed exactly like shutil.rmtree."""
    dump_dir = tmp_path / "dump"
    (dump_dir / "nested").mkdir(parents=True)
    (dump_dir / "nested" / "config.json").write_text("{}", encoding="utf-8")

    remove_module_dump(dump_dir)

    assert not dump_dir.exists()


def test_remove_module_dump_unknown_engine_logs_and_removes(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """An unknown engine cannot be cleaned up in the cluster, but the local dir still goes."""
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    (dump_dir / "remote_manifest.json").write_text(
        json.dumps({"engine": "some-future-engine", "index": "x", "dump_id": "y"}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        remove_module_dump(dump_dir)

    assert not dump_dir.exists()
    assert "some-future-engine" in caplog.text


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker not available; testcontainers cannot boot OpenSearch")
def test_remove_module_dump_deletes_cluster_generation(opensearch_container: tuple[str, int]) -> None:
    """The generation's lifetime is exactly its dump directory's lifetime."""
    host, port = opensearch_container
    backend = _os_backend(host, port, f"test_gen_{uuid.uuid4().hex[:8]}")
    embeddings, documents = _one_hot_docs("best")
    backend.add(embeddings, documents)

    module_dump = Path(tempfile.mkdtemp()) / "module"
    vector_dump = module_dump / "some" / "nesting" / "vector_index"  # helper must scan the tree
    backend.dump(vector_dump)
    generation = json.loads((vector_dump / "remote_manifest.json").read_text(encoding="utf-8"))["index"]

    remove_module_dump(module_dump)

    assert not module_dump.exists()
    assert not backend._client.indices.exists(index=generation)
