from autointent.configs import VectorIndexConfig

from .base_backend import BaseBackend


class OpenSearchBackend(BaseBackend):
    def __init__(self, config: VectorIndexConfig) -> None:
        try:
            import opensearchpy
        except ImportError as e:
            msg = "Unable to create OpenSearch vector index. Install opensearch-py python package first."
            raise RuntimeError(msg) from e

        self.client = opensearchpy.OpenSearch(hosts=config.opensearch.hosts, **config.opensearch.kwargs)
