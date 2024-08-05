import itertools as it
import os

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(
        self, intent_records: os.PathLike, db_path: os.PathLike = "../data/chroma"
    ):
        (
            self.utterances_train,
            self.utterances_test,
            self.labels_train,
            self.labels_test,
        ) = split_sample_utterances(intent_records)

        self.client = PersistentClient(path=db_path)

    def create_collection(
            self,
            model_name: str,
            db_name: str = "example_collection",
            device='cuda'
        ):
        emb_func = SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            trust_remote_code=True,
            device=device
        )
        collection = self.client.get_or_create_collection(
            name=db_name,
            embedding_function=emb_func,
            metadata={"n_classes": len(set(self.labels_train))},
        )
        collection.add(
            documents=self.utterances_train,
            ids=[f"{i}-{db_name}" for i in range(len(self.utterances_train))],
            metadatas=[{"intent_id": lab} for lab in self.labels_train],
        )
        self.collection = collection
        return collection

    def delete_collection(self, db_name: str):
        self.client.delete_collection(db_name)


def get_sample_utterances(dataset: list[dict]):
    """get plain list of all sample utterances and their intent labels"""
    utterances = [intent["sample_utterances"] for intent in dataset]
    labels = [
        [intent["intent_id"]] * len(uts) for intent, uts in zip(dataset, utterances)
    ]

    utterances = list(it.chain.from_iterable(utterances))
    labels = list(it.chain.from_iterable(labels))

    return utterances, labels


def split_sample_utterances(dataset: list[dict]):
    """
    Return: utterances_train, utterances_test, labels_train, labels_test

    TODO: ensure stratified train test splitting (test set must contain all classes)
    """

    utterances, labels = get_sample_utterances(dataset)

    return train_test_split(
        utterances,
        labels,
        test_size=0.25,
        random_state=0,
        stratify=labels,
        shuffle=True,
    )
