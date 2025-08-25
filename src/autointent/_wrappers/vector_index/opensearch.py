import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy._typing import NDArray
from typing_extensions import Self

from autointent.configs import OpenSearchConfig
from autointent.custom_types import Document

from .base_backend import BaseIndexBackend


class OpenSearchBackend(BaseIndexBackend):
    _documents_filename = "documents.json"
    _config_filename = "config.json"
    _embeddings_filename = "embeddings.json"
    _vector_size_filename = "vector_size.txt"

    def __init__(self, config: OpenSearchConfig, vector_size: int) -> None:
        try:
            import opensearchpy  # type: ignore[import-not-found]

            self._opensearchpy = opensearchpy
        except ImportError as e:
            msg = "Unable to create OpenSearch vector index. Install opensearch-py python package first."
            raise RuntimeError(msg) from e

        self.vector_size = vector_size
        self.config = config.model_copy()
        self._client = opensearchpy.OpenSearch(hosts=config.hosts, **config.init_kwargs)
        self._index_name = self.config.index_name

    @property
    def index_name(self) -> str:
        if self._index_name is None:
            msg = "Index is not set. Either use existing collection or add some documents."
            raise RuntimeError(msg)
        return self._index_name

    def _init_index(self) -> None:
        if not self._client.indices.exists(index=self.index_name):
            # Create index for exact vector search using script scoring
            index_body = {
                "settings": {
                    "index": {
                        "knn": False,  # Disable approximate kNN for exact search
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    }
                },
                "mappings": {
                    "properties": {
                        "values": {
                            "type": "knn_vector",
                            "dimension": self.vector_size,
                            # No method specified - this enables exact search with script scoring
                        },
                        "text": {
                            "type": "text",
                            "analyzer": "standard",
                        },
                        "label": {
                            "type": "keyword",
                        },
                    }
                },
            }
            self._client.indices.create(index=self.index_name, body=index_body)

    def clear_ram(self) -> None:
        """Clear the index by deleting all documents."""
        if self._client.indices.exists(index=self.index_name):
            self._client.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}},
                refresh=True,
            )

    def add(self, embeddings: NDArray[Any], documents: list[Document]) -> None:
        """Add embeddings and documents to OpenSearch index."""
        if len(embeddings) != len(documents):
            msg = f"Number of embeddings ({len(embeddings)}) must match number of documents ({len(documents)})"
            raise ValueError(msg)

        if self._index_name is None:
            self._index_name = hashlib.sha256(documents[0].text.encode("utf-8")).hexdigest()[:16]
            self.config.index_name = self._index_name

        # Prepare bulk data
        bulk_data = []
        for i, (embedding, doc) in enumerate(zip(embeddings, documents, strict=True)):
            # Use a unique ID for each document using SHA256 for consistent hashing
            text_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()[:16]
            doc_id = f"{text_hash}_{i}"
            bulk_data.append(
                {
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": {
                        "values": embedding.tolist(),
                        "text": doc.text,
                        "label": doc.label,
                    },
                }
            )

        self._init_index()

        # Use bulk API for efficient indexing
        try:
            success_count, failed_items = self._opensearchpy.helpers.bulk(
                self._client, bulk_data, stats_only=False, raise_on_error=False
            )

        except Exception as e:
            msg = f"Bulk indexing failed: {e}"
            raise RuntimeError(msg) from e

        if failed_items:
            msg = f"Failed to index {len(failed_items)} documents out of {len(bulk_data)}"
            raise RuntimeError(msg)

        # Refresh index to make documents searchable immediately
        # Note: For large datasets, consider batching refreshes or using refresh=wait_for in bulk operations
        self._client.indices.refresh(index=self.index_name)

    def query(self, embedding: NDArray[Any], k: int) -> tuple[NDArray[Any], list[list[Document]]]:
        """Query the index using exact vector similarity search with script scoring."""
        self._validate_embeddings(embedding)

        # Prepare multi-search queries using script_score for exact search
        search_queries: list[dict[str, Any]] = []
        for query_vector in embedding:
            query_body = {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},  # Match all documents first
                        "script": {
                            "lang": "knn",
                            "source": "knn_score",
                            "params": {
                                "field": "values",
                                "query_value": query_vector.tolist(),
                                "space_type": "cosinesimil",  # Use cosine similarity
                            },
                        },
                    }
                },
                "_source": ["text", "label"],
            }
            search_queries.append({"index": self.index_name})
            search_queries.append(query_body)

        # Execute multi-search
        response = self._client.msearch(body=search_queries)

        # Process results
        cosine_similarities = []
        documents = []

        for response_item in response["responses"]:
            if "error" in response_item:
                msg = f"OpenSearch query error: {response_item['error']}"
                raise RuntimeError(msg)

            hits = response_item["hits"]["hits"]

            # Extract similarities (OpenSearch script_score returns exact similarity scores)
            similarities = np.array([hit["_score"] for hit in hits])
            cosine_similarities.append(similarities)

            # Extract documents
            docs = [Document(text=hit["_source"]["text"], label=hit["_source"]["label"]) for hit in hits]
            documents.append(docs)

        return np.array(cosine_similarities), documents

    def get_all_embeddings(self) -> NDArray[Any]:
        """Retrieve all embeddings from the index."""
        # Use scroll API to get all documents
        search_body = {
            "query": {"match_all": {}},
            "_source": ["values"],
            "size": 1000,  # Batch size
        }

        embeddings = []
        response = self._client.search(
            index=self.index_name,
            body=search_body,
            scroll="1m",
        )

        scroll_id = response["_scroll_id"]

        try:
            # Process the first batch of results from the initial search
            hits = response["hits"]["hits"]
            for hit in hits:
                embeddings.append(hit["_source"]["values"])  # noqa: PERF401

            # Continue scrolling through remaining batches
            while True:
                response = self._client.scroll(scroll_id=scroll_id, scroll="1m")
                hits = response["hits"]["hits"]
                if not hits:
                    break

                for hit in hits:
                    embeddings.append(hit["_source"]["values"])  # noqa: PERF401
        finally:
            # Clean up scroll context
            self._client.clear_scroll(scroll_id=scroll_id)

        return np.array(embeddings)

    def dump(self, path: Path) -> None:
        """Save index data to files."""
        path.mkdir(parents=True, exist_ok=True)

        with (path / self._config_filename).open("w", encoding="utf-8") as file:
            json.dump(self.config.model_dump(), file, indent=4, ensure_ascii=False)

        with (path / self._vector_size_filename).open("w", encoding="utf-8") as file:
            file.write(str(self.vector_size))

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load index from saved files."""
        with (path / cls._config_filename).open("r", encoding="utf-8") as file:
            config_data = json.load(file)

        with (path / cls._vector_size_filename).open("r", encoding="utf-8") as file:
            vector_size = int(file.read())

        config = OpenSearchConfig.model_validate(config_data)
        return cls(config=config, vector_size=vector_size)
