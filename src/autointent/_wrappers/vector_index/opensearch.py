from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

from autointent.configs import OpenSearchConfig
from autointent.custom_types import Document

from .base_backend import MANIFEST_FILENAME, BaseIndexBackend

if TYPE_CHECKING:
    from pathlib import Path

    from numpy._typing import NDArray
    from typing_extensions import Self


logger = logging.getLogger(__name__)


class OpenSearchBackend(BaseIndexBackend):
    _config_filename = "config.json"
    _vector_size_filename = "vector_size.txt"
    _manifest_filename = MANIFEST_FILENAME

    def __init__(self, config: OpenSearchConfig, vector_size: int) -> None:
        try:
            import opensearchpy

            self._opensearchpy = opensearchpy
        except ImportError as e:
            msg = "Unable to create OpenSearch vector index. Install opensearch-py python package first."
            raise RuntimeError(msg) from e

        self.vector_size = vector_size
        self.config = config.model_copy()
        self._client = opensearchpy.OpenSearch(hosts=config.hosts, **config.init_kwargs)
        self._index_name = self.config.index_name
        self._has_written = False
        self._generation_index: str | None = None
        self._read_only = False

    @property
    def index_name(self) -> str:
        if self._index_name is None:
            msg = "Index is not set. Either use existing collection or add some documents."
            raise RuntimeError(msg)
        return self._index_name

    def _index_body(self, dump_id: str | None = None) -> dict[str, Any]:
        """Index settings/mappings for exact vector search; optionally stamped with a dump identity."""
        body: dict[str, Any] = {
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
        if dump_id is not None:
            body["mappings"]["_meta"] = {"dump_id": dump_id}
        return body

    def _init_index(self) -> None:
        if not self._client.indices.exists(index=self.index_name):
            self._client.indices.create(index=self.index_name, body=self._index_body())

    def clear_ram(self) -> None:
        """Release local resources (none held): documents live in the remote index and are not touched.

        Deleting remote documents here would destroy the durable state that ``dump()``
        only references — the dumped pipeline would reload an empty index (#342).
        Use :meth:`reset` to actually drop index contents.
        """

    def reset(self) -> None:
        """Drop all documents from the remote index (durable state)."""
        if self._client.indices.exists(index=self.index_name):
            self._client.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}},
                refresh=True,
            )

    def add(self, embeddings: NDArray[Any], documents: list[Document]) -> None:
        """Add embeddings and documents to OpenSearch index.

        The first ``add()`` call on this instance replaces any pre-existing contents of
        the remote index (fit-replaces semantics, see #342); subsequent calls append.
        This also holds for instances restored via ``load()``.
        """
        if self._read_only:
            msg = (
                f"This instance was loaded from a dump and serves the immutable generation index "
                f"'{self.index_name}'. Create a new backend (or re-fit the module) to write data."
            )
            raise RuntimeError(msg)

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

        if not self._has_written:
            self.reset()

        # Use bulk API for efficient indexing
        try:
            _, failed_items = self._opensearchpy.helpers.bulk(
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

        self._has_written = True
        self._generation_index = None  # the live index is the source of truth again

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

            if not hits:
                msg = (
                    f"OpenSearch index '{self.index_name}' returned no documents for a query. "
                    "The index is empty: fit() was never called on it, or it was reset."
                )
                raise RuntimeError(msg)
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

    def _copy_index(self, source: str, dest: str) -> None:
        """Server-side copy: data moves shard-to-shard inside the cluster, never through the client.

        Polls the task API instead of ``wait_for_completion=true`` so arbitrarily large
        corpora are not capped by the HTTP client timeout.
        """
        self._client.indices.refresh(index=source)
        response = self._client.reindex(
            body={"source": {"index": source}, "dest": {"index": dest}},
            wait_for_completion=False,
        )
        task_id = response["task"]
        while True:
            status = self._client.tasks.get(task_id=task_id)
            if status.get("completed"):
                break
            time.sleep(0.2)
        error = status.get("error")
        failures = status.get("response", {}).get("failures", [])
        if error or failures:
            msg = f"Server-side copy from '{source}' to '{dest}' failed: {error or failures}"
            raise RuntimeError(msg)
        self._client.indices.refresh(index=dest)

    def _bind_generation(self, generation: str, dump_id: str) -> None:
        """Bind this instance read-only to a dump generation, verifying it is intact."""
        stored: str | None = None
        if self._client.indices.exists(index=generation):
            mappings = self._client.indices.get_mapping(index=generation)[generation]["mappings"]
            stored = mappings.get("_meta", {}).get("dump_id")
        if stored != dump_id:
            msg = (
                f"dump references cluster index '{generation}' which no longer exists or was recreated "
                f"(expected dump_id={dump_id!r}, found {stored!r})"
            )
            raise RuntimeError(msg)
        self._index_name = generation
        self._generation_index = generation
        self._read_only = True

    @classmethod
    def _delete_generation(cls, client: Any, manifest: dict[str, Any]) -> None:  # noqa: ANN401
        """Delete the generation index a manifest references — only if we still own it."""
        generation = manifest["index"]
        if not client.indices.exists(index=generation):
            return
        mappings = client.indices.get_mapping(index=generation)[generation]["mappings"]
        if mappings.get("_meta", {}).get("dump_id") != manifest["dump_id"]:
            logger.warning(
                "cluster index '%s' was recreated since this dump was written; leaving it in place",
                generation,
            )
            return
        client.indices.delete(index=generation)

    @classmethod
    def delete_dumped_generation(cls, path: Path) -> None:
        """Delete the cluster-side generation referenced by the dump directory at ``path``.

        Reads ``remote_manifest.json`` and ``config.json`` from ``path`` to locate the
        cluster and the index. Safe to call twice; refuses to delete an index whose
        ``_meta.dump_id`` no longer matches the manifest.
        """
        with (path / cls._manifest_filename).open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        with (path / cls._config_filename).open("r", encoding="utf-8") as file:
            config = OpenSearchConfig.model_validate(json.load(file))

        try:
            import opensearchpy
        except ImportError as e:  # same optional dependency story as __init__
            msg = "Unable to delete OpenSearch dump generation. Install opensearch-py python package first."
            raise RuntimeError(msg) from e

        client = opensearchpy.OpenSearch(hosts=config.hosts, **config.init_kwargs)
        cls._delete_generation(client, manifest)

    def dump(self, path: Path) -> None:
        """Snapshot the index into an immutable cluster-side generation.

        Creates ``{base}-best-{uuid}`` with the same mapping as the live index, copies the
        current contents into it server-side, write-blocks it, and records it in
        ``remote_manifest.json`` so ``load()`` serves exactly the data present now — later
        fits of the live index cannot change it (issue #343). If the instance was never
        fitted, only the plain config reference is written.
        """
        path.mkdir(parents=True, exist_ok=True)

        manifest_path = path / self._manifest_filename
        old_manifest: dict[str, Any] | None = None
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as file:
                old_manifest = json.load(file)

        manifest: dict[str, Any] | None = None
        if self._index_name is not None:
            dump_id = uuid.uuid4().hex[:12]
            base = self.index_name.split("-best-")[0]
            generation = f"{base}-best-{dump_id}"
            self._client.indices.create(index=generation, body=self._index_body(dump_id))
            source = self._generation_index or self.index_name
            self._copy_index(source=source, dest=generation)
            self._client.indices.put_settings(index=generation, body={"index": {"blocks": {"write": True}}})
            alias = f"{base}-best"
            actions: list[dict[str, Any]] = [{"add": {"index": generation, "alias": alias}}]
            existing = self._client.indices.get_alias(name=alias, ignore=404)
            if isinstance(existing, dict) and "error" not in existing:
                actions = [{"remove": {"index": index, "alias": alias}} for index in existing] + actions
            self._client.indices.update_aliases(body={"actions": actions})  # remove+add is atomic
            self._generation_index = generation
            manifest = {"engine": "opensearch", "index": generation, "dump_id": dump_id}

        with (path / self._config_filename).open("w", encoding="utf-8") as file:
            json.dump(self.config.model_dump(), file, indent=4, ensure_ascii=False)

        with (path / self._vector_size_filename).open("w", encoding="utf-8") as file:
            file.write(str(self.vector_size))

        if manifest is not None:
            with manifest_path.open("w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=4, ensure_ascii=False)
        elif old_manifest is not None:
            manifest_path.unlink()

        if old_manifest is not None and (manifest is None or old_manifest["index"] != manifest["index"]):
            self._delete_generation(self._client, old_manifest)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load index from saved files.

        If the dump carries a ``remote_manifest.json``, the instance binds read-only to the
        immutable generation index recorded there (and verifies its identity). Dumps without
        a manifest (pre-#343, or dumped before any fit) bind to the configured live index.
        """
        with (path / cls._config_filename).open("r", encoding="utf-8") as file:
            config_data = json.load(file)

        with (path / cls._vector_size_filename).open("r", encoding="utf-8") as file:
            vector_size = int(file.read())

        config = OpenSearchConfig.model_validate(config_data)
        instance = cls(config=config, vector_size=vector_size)

        manifest_path = path / cls._manifest_filename
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
            instance._bind_generation(manifest["index"], manifest["dump_id"])
        return instance
