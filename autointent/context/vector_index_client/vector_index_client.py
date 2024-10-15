import logging

from autointent.context.data_handler import DataHandler

from .vector_index import VectorIndex


class VectorIndexClient:
    def __init__(self, device: str, multilabel: bool, n_classes: int) -> None:
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.multilabel = multilabel
        self.n_classes = n_classes
        self.indexes: dict[str, VectorIndex] = {}
        self.model_name = None

    def set_best_embedder_name(self, model_name: str) -> None:
        if model_name not in self.indexes:
            msg = f"model {model_name} wasn't created before"
            self._logger.error(msg)
            raise ValueError(msg)

        self.model_name = model_name

    def create_index(self, model_name: str, data_handler: DataHandler) -> VectorIndex:
        self._logger.info("Creating index for model: %s", model_name)

        index = VectorIndex(model_name, self.device)
        index.add(data_handler.utterances_train, data_handler.labels_train)

        self.indexes[model_name] = index

        return index

    def delete_index(self, model_name: str) -> None:
        if model_name in self.indexes:
            self._logger.debug("Deleting index for model: %s", model_name)
            self.indexes[model_name].delete()
            del self.indexes[model_name]

    def get_index(self, model_name: str) -> VectorIndex:
        return self.indexes[model_name]
