import os
import logging
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .data_handler import DataHandler

logger = logging.getLogger(__name__)

class VectorIndex:
    def __init__(self, db_dir: os.PathLike, device: str, multilabel: bool, n_classes):
        self.device = device
        self.multilabel = multilabel
        self.n_classes = n_classes

        settings = Settings(
            chroma_segment_cache_policy="LRU",
            chroma_memory_limit_bytes=2 * 1024 * 1024 * 1024,  # 2 GB
        )
        self.client = PersistentClient(path=db_dir, settings=settings)

    def get_collection(self, model_name: str, device=None):
        device = device if device is not None else self.device
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

        if self.multilabel:
            metadatas = multilabel_labels_as_metadata(data_handler.labels_train)
        else:
            metadatas = multiclass_labels_as_metadata(data_handler.labels_train)

        existing_ids = set(collection.get()["ids"])
        new_documents = []
        new_ids = []
        new_metadatas = []

        for i, (utterance, metadata) in enumerate(zip(data_handler.utterances_train, metadatas)):
            new_id = f"{i}-{db_name}"
            if new_id not in existing_ids:
                new_documents.append(utterance)
                new_ids.append(new_id)
                new_metadatas.append(metadata)

        if new_documents:
            collection.add(
                documents=new_documents,
                ids=new_ids,
                metadatas=new_metadatas,
            )
        return collection

    def delete_collection(self, model_name: str):
        db_name = model_name.replace("/", "_")
        self.client.delete_collection(db_name)

    def print_info(self):
        logger.info("VectorIndex Information:")
        logger.info(f"Device: {self.device}")
        logger.info(f"Multilabel: {self.multilabel}")
        logger.info(f"Number of classes: {self.n_classes}")

        collections = self.client.list_collections()
        logger.info(f"Number of collections: {len(collections)}")

        for collection in collections:
            collection_name = collection.name
            collection_obj = self.client.get_collection(collection_name)
            count = collection_obj.count()
            metadata = collection_obj.metadata
            logger.info(f"Collection: {collection_name}")
            logger.info(f"  Number of items: {count}")
            logger.info(f"  Metadata: {metadata}")

            # Примеры данных (если есть)
            if count > 0:
                sample = collection_obj.get(limit=1)
                logger.info("  Sample item:")
                logger.info(f"    ID: {sample['ids'][0]}")
                logger.info(f"    Document: {sample['documents'][0]}")
                logger.info(f"    Metadata: {sample['metadatas'][0]}")


def multiclass_labels_as_metadata(labels_list: list[int]):
    return [{"intent_id": lab} for lab in labels_list]


def multilabel_labels_as_metadata(labels_list: list[list[int]]):
    """labels_list is already in binary format"""
    return [{str(i): lab for i, lab in enumerate(labs)} for labs in labels_list]


def multiclass_metadata_as_labels(metadata: list[dict]):
    return [dct["intent_id"] for dct in metadata]


def multilabel_metadata_as_labels(metadata: list[dict], n_classes):
    return [[dct[str(i)] for i in range(n_classes)] for dct in metadata]
