import tempfile
from pathlib import Path

import numpy as np
import pytest

from autointent import VectorIndex
from autointent.configs import EmbedderConfig, FaissConfig, OpenSearchConfig
from autointent.custom_types import Document

# Check if opensearch-py is available
opensearch_available = True
try:
    import opensearchpy
except ImportError:
    opensearch_available = False


def is_opensearch_running() -> bool:
    """Check if OpenSearch is running on localhost:9200."""
    if not opensearch_available:
        return False

    try:
        client = opensearchpy.OpenSearch(hosts=[{"host": "localhost", "port": 9200}], timeout=1, max_retries=0)
        client.cluster.health(timeout=1)
    except opensearchpy.ConnectionError as e:
        if "Connection refused" in str(e):
            return False
        raise
    else:
        return True


# Backend configurations for parametrization
backend_configs = [
    pytest.param(FaissConfig(), id="faiss"),
    pytest.param(
        OpenSearchConfig(
            hosts=[{"host": "localhost", "port": 9200}],
            index_name=None,  # Will be auto-generated
        ),
        marks=pytest.mark.skipif(
            not opensearch_available or not is_opensearch_running(),
            reason="OpenSearch not available or not running on localhost:9200",
        ),
        id="opensearch",
    ),
]


@pytest.mark.parametrize("vector_config", backend_configs)
class TestVectorIndex:
    """Unified test class for VectorIndex with different backends."""

    @pytest.fixture
    def embedder_config(self) -> EmbedderConfig:
        """Create a lightweight embedder config for testing."""
        return EmbedderConfig.from_search_config("sentence-transformers/all-MiniLM-L6-v2")

    @pytest.fixture
    def vector_index(self, embedder_config: EmbedderConfig, vector_config) -> VectorIndex:
        """Create a VectorIndex instance for testing."""
        # For OpenSearch, ensure unique index names to avoid test interference
        if isinstance(vector_config, OpenSearchConfig):
            import uuid

            unique_id = str(uuid.uuid4())[:8]
            vector_config.index_name = f"test_index_{unique_id}"

        return VectorIndex(embedder_config=embedder_config, config=vector_config)

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

    def test_initialization(self, vector_index: VectorIndex, vector_config):
        """Test VectorIndex initialization."""
        assert vector_index.config == vector_config
        assert hasattr(vector_index, "embedder")
        # Index should not be created until first add
        assert not hasattr(vector_index, "index")

    def test_add_texts_and_labels(self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]):
        """Test adding texts and labels to the index."""
        vector_index.add(sample_texts, sample_labels)

        # Index should be created after first add
        assert hasattr(vector_index, "index")

        # Check that embeddings were stored
        embeddings = vector_index.get_all_embeddings()
        assert embeddings.shape[0] == len(sample_texts)
        assert embeddings.shape[1] > 0  # Should have some dimensions

    def test_add_multiple_batches(self, vector_index: VectorIndex):
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

    def test_query_by_text(self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]):
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

    def test_query_by_embedding(self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]):
        """Test querying the index with embeddings."""
        vector_index.add(sample_texts, sample_labels)

        # Get embeddings for query
        query_embeddings = vector_index.embedder.embed(["How to reset password?"], "query")

        distances, documents = vector_index.query(query_embeddings, k=3)

        assert len(distances) == 1  # One query
        assert len(documents) == 1
        assert len(distances[0]) == 3  # k=3 neighbors
        assert len(documents[0]) == 3

    def test_query_empty_index_raises_error(self, vector_index: VectorIndex):
        """Test that querying an empty index raises an error."""
        with pytest.raises(ValueError, match="Index is not created yet"):
            vector_index.get_all_embeddings()

    def test_query_with_k_larger_than_index(self, vector_index: VectorIndex):
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

    def test_clear_ram(self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]):
        """Test clearing the index from RAM."""
        vector_index.add(sample_texts, sample_labels)

        # Clear RAM
        vector_index.clear_ram()

        # For FaissBackend, index should be reset
        # For OpenSearchBackend, documents should be deleted
        # Both should handle this gracefully
        if isinstance(vector_index.config, FaissConfig):
            # Faiss index should be reset but still exist
            assert hasattr(vector_index, "index")
            embeddings = vector_index.get_all_embeddings()
            assert embeddings.shape[0] == 0

    def test_dump_and_load(self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]):
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
    ):
        """Test loading with embedder config override."""
        vector_index.add(sample_texts, sample_labels)

        with tempfile.TemporaryDirectory() as temp_dir:
            dump_path = Path(temp_dir)
            vector_index.dump(dump_path)

            # Create override config
            override_config = EmbedderConfig.from_search_config("sentence-transformers/all-MiniLM-L6-v2")
            override_config.device = "cpu"
            override_config.batch_size = 1

            # Load with override
            loaded_index = VectorIndex.load(dump_path, embedder_override_config=override_config)

            # Check that override was applied
            assert loaded_index.embedder.config.device == "cpu"
            assert loaded_index.embedder.config.batch_size == 1

    def test_error_handling_mismatched_lengths(self, vector_index: VectorIndex):
        """Test error handling when texts and labels have different lengths."""
        texts = ["Hello", "World"]
        labels = [0, 1, 2]  # Wrong length

        with pytest.raises(ValueError, match="mismatch"):
            vector_index.add(texts, labels)

    def test_backend_specific_behavior(
        self, vector_index: VectorIndex, sample_texts: list[str], sample_labels: list[int]
    ):
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
            # Index name should be auto-generated if not provided
            assert vector_index.index.index_name is not None


class TestVectorIndexEdgeCases:
    """Test edge cases and error conditions."""

    def test_abstract_config_raises_error(self):
        """Test that using abstract VectorIndexConfig raises an error."""
        from autointent.configs import VectorIndexConfig

        embedder_config = EmbedderConfig.from_search_config("sentence-transformers/all-MiniLM-L6-v2")

        vector_index = VectorIndex(embedder_config=embedder_config, config=VectorIndexConfig())
        with pytest.raises(TypeError, match="Passed abstract vector index config"):
            vector_index.add(["test"], [0])

    def test_opensearch_dependency_error(self, monkeypatch):
        """Test OpenSearch dependency error handling."""
        # Mock opensearchpy import to fail
        import sys

        original_modules = sys.modules.copy()

        try:
            # Remove opensearchpy from sys.modules if it exists
            if "opensearchpy" in sys.modules:
                del sys.modules["opensearchpy"]

            # Mock import to raise ImportError
            def mock_import(name, *args, **kwargs):  # noqa: ARG001
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
