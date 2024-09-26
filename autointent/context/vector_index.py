import os
import logging

from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .data_handler import DataHandler


class VectorIndex:
    def __init__(self, db_dir: os.PathLike, device: str, multilabel: bool, n_classes: int):
        self._logger = logging.getLogger(__name__)

        self.device = device
        self.multilabel = multilabel
        self.n_classes = n_classes

        self._logger.debug("connecting to Chroma DB client...")
        settings = Settings(
            chroma_segment_cache_policy="LRU",
            chroma_memory_limit_bytes=2 * 1024 * 1024 * 1024,  # 2 GB
        )
        self.client = PersistentClient(path=db_dir, settings=settings)

    def get_collection(self, model_name: str, device=None):
        device = device if device is not None else self.device
        self._logger.info(f"spawning sentence transformer instance of {model_name} on {device=}...")
        emb_func = SentenceTransformerEmbeddingFunction(
            model_name=model_name, trust_remote_code=True, device=device, tokenizer_kwargs=dict(truncation=True)
        )
        db_name = model_name.replace("/", "_")
        collection = self.client.get_or_create_collection(
            name=db_name,
            embedding_function=emb_func,
            metadata=dict(multilabel=self.multilabel, n_classes=self.n_classes) | {"hnsw:space": "cosine"},
        )
        return collection

    def create_collection(self, model_name: str, data_handler: DataHandler, device=None):
        device = device if device is not None else self.device

        collection = self.get_collection(model_name, device)
        db_name = model_name.replace("/", "_")

        metadatas = self.labels_as_metadata(data_handler.labels_train)

        self._logger.debug("adding train utterances to vector index...")
        collection.add(
            documents=data_handler.utterances_train,
            ids=[f"{i}-{db_name}" for i in range(len(data_handler.utterances_train))],
            metadatas=metadatas,
        )
        return collection

    def delete_collection(self, model_name: str):
        self._logger.debug(f"deleting collection for {model_name}...")
        db_name = model_name.replace("/", "_")
        self.client.delete_collection(db_name)
    
    def metadata_as_labels(self, metadata: list[dict]):
        if self.multilabel:
            return _multilabel_metadata_as_labels(metadata, self.n_classes)
        return _multiclass_metadata_as_labels(metadata)

    def labels_as_metadata(self, metadata: list[dict]):
        if self.multilabel:
            return _multilabel_labels_as_metadata(metadata)
        return _multiclass_labels_as_metadata(metadata)


def _multiclass_labels_as_metadata(labels_list: list[int]):
        return [{"intent_id": lab} for lab in labels_list]

def _multilabel_labels_as_metadata(labels_list: list[list[int]]):
    """labels_list is already in binary format"""
    return [{str(i): lab for i, lab in enumerate(labs)} for labs in labels_list]

def _multiclass_metadata_as_labels(metadata: list[dict]):
    return [dct["intent_id"] for dct in metadata]

def _multilabel_metadata_as_labels(metadata: list[dict], n_classes):
    return [[dct[str(i)] for i in range(n_classes)] for dct in metadata]
