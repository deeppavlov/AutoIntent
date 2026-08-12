from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from autointent import VectorIndex
from autointent._wrappers.vector_index.faiss import FaissBackend
from autointent._wrappers.vector_index.opensearch import OpenSearchBackend
from autointent.configs import (
    FaissConfig,
    HashingVectorizerEmbeddingConfig,
    OpenSearchConfig,
    TaskTypeEnum,
    VectorIndexConfig,
)
from autointent.custom_types import Document
from tests.conftest import get_test_embedder_config

if TYPE_CHECKING:
    from types import ModuleType


def _docker_available() -> bool:
    """Detect whether Docker is reachable for testcontainers (skipped on most Windows CI)."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


_DOCKER_AVAILABLE = _docker_available()

# Backend configurations for parametrization.
# vector_config is heterogeneous: FaissConfig is a concrete config; "opensearch_lazy" is a sentinel
# string resolved inside the vector_index fixture to a real OpenSearchConfig (it needs the
# session-scoped opensearch_container fixture, which can't be constructed at parametrize-collect time).
backend_configs = [
    pytest.param(FaissConfig(), id="faiss"),
    pytest.param(
        "opensearch_lazy",
        id="opensearch",
        marks=pytest.mark.skipif(
            not _DOCKER_AVAILABLE,
            reason="Docker not available; testcontainers cannot boot OpenSearch",
        ),
    ),  # resolved to real config in vector_index fixture
]


@pytest.mark.parametrize("vector_config", backend_configs)
class TestVectorIndex:
    """Unified test class for VectorIndex with different backends."""

    @pytest.fixture
    def embedder_config(self) -> HashingVectorizerEmbeddingConfig:
        """Create a lightweight embedder config for testing."""
        return get_test_embedder_config()

    @pytest.fixture
    def vector_index(
        self,
        embedder_config: HashingVectorizerEmbeddingConfig,
        vector_config: FaissConfig | str,
        request: pytest.FixtureRequest,
    ) -> VectorIndex:
        """Create a VectorIndex instance for testing."""
        resolved_config: FaissConfig | OpenSearchConfig
        if isinstance(vector_config, str) and vector_config == "opensearch_lazy":
            host, port = request.getfixturevalue("opensearch_container")
            unique_id = str(uuid.uuid4())[:8]
            resolved_config = OpenSearchConfig(
                hosts=[{"host": host, "port": port}],
                index_name=f"test_index_{unique_id}",
            )
        else:
            assert isinstance(vector_config, FaissConfig)
            resolved_config = vector_config

        return VectorIndex(embedder_config=embedder_config, config=resolved_config)

    @pytest.fixture
    def sample_texts(self) -> list[str]:
        """Sample texts for testing."""
        return [
            "How do I reset my password?",
            "What are your business hours?",
            "Can I change my account settings?",
            "Where is my order?",
            "How do I contact customer support?",
        ]

    @pytest.fixture
    def sample_labels(self) -> list[int]:
        """Sample labels corresponding to texts."""
        return [0, 1, 0, 2, 1]

    def test_initialization(self, vector_index: VectorIndex, vector_config: FaissConfig | str) -> None:
        """Test VectorIndex initialization."""
        if isinstance(vector_config, str):  # placeholder for opensearch
            assert isinstance(vector_index.config, OpenSearchConfig)
        else:
            assert vector_index.config == vector_config
        assert hasattr(vector_index, "embedder")
        assert not hasattr(vector_index, "index")

    def test_add_texts_and_labels(
        self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]
    ) -> None:
        """Test adding texts and labels to the index."""
        vector_index.add(sample_texts, sample_labels)

        # Index should be created after first add
        assert hasattr(vector_index, "index")

        # Check that embeddings were stored
        embeddings = vector_index.get_all_embeddings()
        assert embeddings.shape[0] == len(sample_texts)
        assert embeddings.shape[1] > 0  # Should have some dimensions

    def test_add_multiple_batches(self, vector_index: VectorIndex) -> None:
        """Test adding multiple batches of data."""
        # Add first batch
        texts1 = ["Hello world", "Good morning"]
        labels1 = [0, 1]
        vector_index.add(texts1, labels1)

        # Add second batch
        texts2 = ["Good evening", "Hello there"]
        labels2 = [1, 0]
        vector_index.add(texts2, labels2)

        # Check total embeddings
        embeddings = vector_index.get_all_embeddings()
        assert embeddings.shape[0] == 4

    def test_first_add_of_new_instance_replaces_index_contents(
        self,
        vector_index: VectorIndex,
        embedder_config: HashingVectorizerEmbeddingConfig,
        sample_texts: list[str],
        sample_labels: list[int],
    ) -> None:
        """A fresh instance's first add() starts from an empty index (issue #342, CV fold isolation).

        Mirrors cross-validation: each fold's fit() constructs a new VectorIndex over the
        same backend config. The previous fold's documents must not survive into this one.
        """
        vector_index.add(sample_texts, sample_labels)

        second_index = VectorIndex(embedder_config=embedder_config, config=vector_index.config)
        fold_texts, fold_labels = sample_texts[1:4], sample_labels[1:4]
        second_index.add(fold_texts, fold_labels)

        assert second_index.get_all_embeddings().shape[0] == len(fold_texts)

    def test_query_by_text(self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]) -> None:
        """Test querying the index with text."""
        vector_index.add(sample_texts, sample_labels)

        # Query with similar text
        query_texts = ["How to reset password?", "What are the hours?"]
        distances, documents = vector_index.query(query_texts, k=2)

        assert len(distances) == 2  # Two queries
        assert len(documents) == 2
        assert len(distances[0]) == 2  # k=2 neighbors
        assert len(documents[0]) == 2

        # Check that documents are returned correctly
        for doc_list in documents:
            for doc in doc_list:
                assert isinstance(doc, Document)
                assert hasattr(doc, "text")
                assert hasattr(doc, "label")

    def test_query_by_embedding(
        self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]
    ) -> None:
        """Test querying the index with embeddings."""
        vector_index.add(sample_texts, sample_labels)

        # Get embeddings for query
        query_embeddings = vector_index.embedder.embed(["How to reset password?"], TaskTypeEnum.query)

        distances, documents = vector_index.query(query_embeddings, k=3)

        assert len(distances) == 1  # One query
        assert len(documents) == 1
        assert len(distances[0]) == 3  # k=3 neighbors
        assert len(documents[0]) == 3

    def test_query_empty_index_raises_error(self, vector_index: VectorIndex) -> None:
        """Test that querying an empty index raises an error."""
        with pytest.raises(ValueError, match="Index is not created yet"):
            vector_index.get_all_embeddings()

    def test_query_with_k_larger_than_index(self, vector_index: VectorIndex) -> None:
        """Test querying with k larger than the number of indexed documents."""
        texts = ["Hello world"]
        labels = [0]
        vector_index.add(texts, labels)

        # Query with k=5 when only 1 document exists
        distances, documents = vector_index.query(["Hello"], k=5)

        # Different backends handle this differently:
        # - Faiss returns k results with padding values for missing documents
        # - OpenSearch would return only available documents
        if isinstance(vector_index.config, FaissConfig):
            # Faiss returns k results but with padding for missing docs
            assert len(distances[0]) == 5
            assert len(documents[0]) == 5
            # The first result should be meaningful, others should be padding
            assert distances[0][0] > -1000  # Real similarity score
            # Check that some results are padding (very negative values)
            assert any(dist < -1000 for dist in distances[0][1:])
        else:
            # OpenSearch and others should return only available documents
            assert len(distances[0]) <= 1
            assert len(documents[0]) <= 1

    def test_clear_ram(self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]) -> None:
        """clear_ram() releases local resources and must not destroy durable state (issue #342)."""
        vector_index.add(sample_texts, sample_labels)

        vector_index.clear_ram()

        if isinstance(vector_index.config, FaissConfig):
            # everything is local: the in-RAM vectors are dropped
            assert vector_index.get_all_embeddings().shape[0] == 0
        else:
            # documents live remotely; releasing local resources must not delete them
            assert vector_index.get_all_embeddings().shape[0] == len(sample_texts)

    def test_dump_survives_clear_ram(
        self,
        vector_index: VectorIndex,
        sample_texts: list[str],
        sample_labels: list[int],
        tmp_path: Path,
    ) -> None:
        """The optimizer dumps the best module, then clear_ram()s it; the dump must stay servable (issue #342)."""
        vector_index.add(sample_texts, sample_labels)
        dump_dir = tmp_path / "dump"
        vector_index.dump(dump_dir)

        vector_index.clear_ram()

        loaded = VectorIndex.load(dump_dir)
        _distances, documents = loaded.query(["password reset"], k=2)
        assert len(documents[0]) == 2

    def test_reset_drops_all_documents(
        self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]
    ) -> None:
        """reset() drops every indexed document, including durable state (issue #342)."""
        vector_index.add(sample_texts, sample_labels)

        vector_index.index.reset()

        assert vector_index.get_all_embeddings().shape[0] == 0
        if isinstance(vector_index.index, FaissBackend):
            # reset() must clear the documents store too, not only the vectors
            assert vector_index.index._documents == []

    def test_dump_and_load(self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]) -> None:
        """Test dumping and loading the vector index."""
        vector_index.add(sample_texts, sample_labels)

        # Test query before dump
        original_distances, original_documents = vector_index.query(["password reset"], k=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            dump_path = Path(temp_dir)

            # Dump the index
            vector_index.dump(dump_path)

            # Load the index
            loaded_index = VectorIndex.load(dump_path)

            # Test that loaded index works the same
            loaded_distances, loaded_documents = loaded_index.query(["password reset"], k=2)

            # Distances should be similar (allowing for small floating point differences)
            np.testing.assert_allclose(original_distances, loaded_distances, rtol=1e-5)

            # Documents should be identical
            assert len(loaded_documents) == len(original_documents)
            for orig_docs, loaded_docs in zip(original_documents, loaded_documents, strict=False):
                assert len(orig_docs) == len(loaded_docs)
                for orig_doc, loaded_doc in zip(orig_docs, loaded_docs, strict=False):
                    assert orig_doc.text == loaded_doc.text
                    assert orig_doc.label == loaded_doc.label

    def test_load_with_embedder_override(
        self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]
    ) -> None:
        """Test loading with embedder config override."""
        vector_index.add(sample_texts, sample_labels)

        with tempfile.TemporaryDirectory() as temp_dir:
            dump_path = Path(temp_dir)
            vector_index.dump(dump_path)

            # Create override config
            override_config = get_test_embedder_config()
            override_config.analyzer = "char"

            # Load with override
            loaded_index = VectorIndex.load(dump_path, embedder_override_config=override_config)

            # Check that loaded index works with overridden config.
            # EmbedderConfig is a union; n_features is on HashingVectorizerEmbeddingConfig only.
            assert isinstance(loaded_index.embedder.config, HashingVectorizerEmbeddingConfig)
            assert loaded_index.embedder.config.n_features == 512

    def test_error_handling_mismatched_lengths(self, vector_index: VectorIndex) -> None:
        """Test error handling when texts and labels have different lengths."""
        texts = ["Hello", "World"]
        labels = [0, 1, 2]  # Wrong length

        with pytest.raises(ValueError, match="mismatch"):
            vector_index.add(texts, labels)

    def test_backend_specific_behavior(
        self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]
    ) -> None:
        """Test backend-specific behavior differences."""
        vector_index.add(sample_texts, sample_labels)

        if isinstance(vector_index.config, FaissConfig):
            # Faiss should have an internal _index attribute
            assert hasattr(vector_index.index, "_index")
            assert hasattr(vector_index.index, "_documents")

        elif isinstance(vector_index.config, OpenSearchConfig):
            # OpenSearch should have a client and index name
            assert hasattr(vector_index.index, "_client")
            assert hasattr(vector_index.index, "_index_name")
            # Index name should be auto-generated if not provided.
            # vector_index.index is typed as BaseIndexBackend, but OpenSearchConfig implies
            # the concrete OpenSearchBackend (which has the index_name property).
            assert isinstance(vector_index.index, OpenSearchBackend)
            assert vector_index.index.index_name is not None


class TestVectorIndexEdgeCases:
    """Test edge cases and error conditions."""

    def test_abstract_config_raises_error(self) -> None:
        """Test that using abstract VectorIndexConfig raises an error."""
        embedder_config = get_test_embedder_config()

        vector_index = VectorIndex(embedder_config=embedder_config, config=VectorIndexConfig())
        with pytest.raises(TypeError, match="Passed abstract vector index config"):
            vector_index.add(["test"], [0])

    def test_opensearch_dependency_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test OpenSearch dependency error handling."""
        # Mock opensearchpy import to fail

        original_modules = sys.modules.copy()

        try:
            # Remove opensearchpy from sys.modules if it exists
            if "opensearchpy" in sys.modules:
                del sys.modules["opensearchpy"]

            # Mock import to raise ImportError
            def mock_import(name: str, *args: object, **kwargs: object) -> ModuleType | None:
                if name == "opensearchpy":
                    msg = "No module named opensearchpy"
                    raise ImportError(msg)
                return original_modules.get(name)

            monkeypatch.setattr("builtins.__import__", mock_import)

            # Import the backend module fresh to trigger the import error
            from autointent._wrappers.vector_index.opensearch import OpenSearchBackend

            config = OpenSearchConfig(hosts=[{"host": "localhost", "port": 9200}])

            with pytest.raises(RuntimeError, match="Install opensearch-py python package first"):
                OpenSearchBackend(config=config, vector_size=384)

        finally:
            # Restore original modules
            sys.modules.update(original_modules)


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker not available; testcontainers cannot boot OpenSearch")
def test_opensearch_empty_index_query_raises_actionable_error(opensearch_container: tuple[str, int]) -> None:
    """Querying an empty index names the problem instead of failing later with a numpy cast error (issue #342)."""
    host, port = opensearch_container
    config = OpenSearchConfig(
        hosts=[{"host": host, "port": port}],
        index_name=f"test_empty_{uuid.uuid4().hex[:8]}",
    )
    backend = OpenSearchBackend(config=config, vector_size=8)
    backend._init_index()

    with pytest.raises(RuntimeError, match="empty"):
        backend.query(np.zeros((1, 8)), k=3)


def _os_backend(host: str, port: int, index_name: str | None, vector_size: int = 8) -> OpenSearchBackend:
    config = OpenSearchConfig(hosts=[{"host": host, "port": port}], index_name=index_name)
    return OpenSearchBackend(config=config, vector_size=vector_size)


def _one_hot_docs(prefix: str, n: int = 4, label: int = 0) -> tuple[np.ndarray, list[Document]]:
    """One-hot embeddings make nearest-neighbor assertions exact: query eye[i] -> doc i."""
    return np.eye(8, dtype="float32")[:n], [Document(text=f"{prefix} {i}", label=label) for i in range(n)]


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker not available; testcontainers cannot boot OpenSearch")
def test_opensearch_dump_creates_write_blocked_generation(opensearch_container: tuple[str, int]) -> None:
    """dump() copies the live index into an immutable generation and records a manifest (issue #343)."""
    import opensearchpy

    host, port = opensearch_container
    live_name = f"test_gen_{uuid.uuid4().hex[:8]}"
    backend = _os_backend(host, port, live_name)
    embeddings, documents = _one_hot_docs("best")
    backend.add(embeddings, documents)

    dump_dir = Path(tempfile.mkdtemp()) / "vector_index"
    backend.dump(dump_dir)

    manifest = json.loads((dump_dir / "remote_manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine"] == "opensearch"
    generation = manifest["index"]
    assert generation.startswith(f"{live_name}-best-")
    assert generation.endswith(manifest["dump_id"])

    client = backend._client
    assert client.indices.exists(index=generation)
    assert client.count(index=generation)["count"] == 4

    meta = client.indices.get_mapping(index=generation)[generation]["mappings"]["_meta"]
    assert meta["dump_id"] == manifest["dump_id"]

    with pytest.raises(opensearchpy.exceptions.TransportError):
        client.index(index=generation, body={"values": [0.0] * 8, "text": "stray write", "label": 0})

    # the pre-existing reference files are still written
    assert (dump_dir / "config.json").exists()
    assert (dump_dir / "vector_size.txt").exists()


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker not available; testcontainers cannot boot OpenSearch")
def test_opensearch_dump_before_fit_writes_no_manifest(opensearch_container: tuple[str, int]) -> None:
    """A never-fitted backend dumps a plain reference (no generation to copy)."""
    host, port = opensearch_container
    backend = _os_backend(host, port, index_name=None)

    dump_dir = Path(tempfile.mkdtemp()) / "vector_index"
    backend.dump(dump_dir)

    assert not (dump_dir / "remote_manifest.json").exists()
    assert (dump_dir / "config.json").exists()


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker not available; testcontainers cannot boot OpenSearch")
def test_opensearch_dumped_pipeline_survives_later_writes(opensearch_container: tuple[str, int]) -> None:
    """THE regression test for issue #343: a dump serves the data present at dump time,
    even after later trials rewrite — or someone deletes — the live index."""
    host, port = opensearch_container
    live_name = f"test_gen_{uuid.uuid4().hex[:8]}"

    best = _os_backend(host, port, live_name)
    embeddings, documents = _one_hot_docs("best")
    best.add(embeddings, documents)
    dump_dir = Path(tempfile.mkdtemp()) / "vector_index"
    best.dump(dump_dir)

    later = _os_backend(host, port, live_name)  # next trial: fresh instance, fit-replaces
    later_embeddings, later_documents = _one_hot_docs("later", label=1)
    later.add(later_embeddings, later_documents)

    best._client.indices.delete(index=live_name)  # even destroying the live index is fine

    loaded = OpenSearchBackend.load(dump_dir)
    _, results = loaded.query(np.eye(8, dtype="float32")[:1], k=1)
    assert results[0][0].text == "best 0"


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker not available; testcontainers cannot boot OpenSearch")
def test_opensearch_load_raises_when_generation_missing(opensearch_container: tuple[str, int]) -> None:
    host, port = opensearch_container
    backend = _os_backend(host, port, f"test_gen_{uuid.uuid4().hex[:8]}")
    embeddings, documents = _one_hot_docs("best")
    backend.add(embeddings, documents)
    dump_dir = Path(tempfile.mkdtemp()) / "vector_index"
    backend.dump(dump_dir)

    generation = json.loads((dump_dir / "remote_manifest.json").read_text(encoding="utf-8"))["index"]
    backend._client.indices.delete(index=generation)

    with pytest.raises(RuntimeError, match="no longer exists or was recreated"):
        OpenSearchBackend.load(dump_dir)


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker not available; testcontainers cannot boot OpenSearch")
def test_opensearch_load_raises_when_generation_recreated(opensearch_container: tuple[str, int]) -> None:
    """A same-named index without our dump_id is somebody else's index — refuse, don't serve it."""
    host, port = opensearch_container
    backend = _os_backend(host, port, f"test_gen_{uuid.uuid4().hex[:8]}")
    embeddings, documents = _one_hot_docs("best")
    backend.add(embeddings, documents)
    dump_dir = Path(tempfile.mkdtemp()) / "vector_index"
    backend.dump(dump_dir)

    generation = json.loads((dump_dir / "remote_manifest.json").read_text(encoding="utf-8"))["index"]
    backend._client.indices.delete(index=generation)
    backend._client.indices.create(index=generation)  # recreated, no _meta.dump_id

    with pytest.raises(RuntimeError, match="no longer exists or was recreated"):
        OpenSearchBackend.load(dump_dir)


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker not available; testcontainers cannot boot OpenSearch")
def test_opensearch_loaded_instance_rejects_writes(opensearch_container: tuple[str, int]) -> None:
    host, port = opensearch_container
    backend = _os_backend(host, port, f"test_gen_{uuid.uuid4().hex[:8]}")
    embeddings, documents = _one_hot_docs("best")
    backend.add(embeddings, documents)
    dump_dir = Path(tempfile.mkdtemp()) / "vector_index"
    backend.dump(dump_dir)

    loaded = OpenSearchBackend.load(dump_dir)
    with pytest.raises(RuntimeError, match="immutable"):
        loaded.add(embeddings, documents)


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker not available; testcontainers cannot boot OpenSearch")
def test_opensearch_manifestless_dump_loads_as_reference(opensearch_container: tuple[str, int]) -> None:
    """Backward compatibility: dumps created before #343 (no manifest) keep today's semantics."""
    host, port = opensearch_container
    live_name = f"test_gen_{uuid.uuid4().hex[:8]}"
    backend = _os_backend(host, port, live_name)
    embeddings, documents = _one_hot_docs("best")
    backend.add(embeddings, documents)
    dump_dir = Path(tempfile.mkdtemp()) / "vector_index"
    backend.dump(dump_dir)
    (dump_dir / "remote_manifest.json").unlink()  # simulate a pre-#343 dump

    loaded = OpenSearchBackend.load(dump_dir)
    assert loaded.index_name == live_name
    _, results = loaded.query(np.eye(8, dtype="float32")[:1], k=1)
    assert results[0][0].text == "best 0"
