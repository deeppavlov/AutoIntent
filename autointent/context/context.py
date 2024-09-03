import numpy as np
from .data_handler import DataHandler
from .optimization_logs import OptimizationLogs
from .vector_index import VectorIndex


class Context:
    def __init__(self, intent_records, device, multilabel, db_dir) -> None:
        self.data_handler = DataHandler(intent_records, multilabel)
        self.optimization_logs = OptimizationLogs()
        self.vector_index = VectorIndex(db_dir, device, multilabel, self.data_handler.n_classes)

        self.device = device
        self.multilabel = multilabel
        self.n_classes = self.data_handler.n_classes

    def get_best_collection(self):
        model_name = self.optimization_logs.get_best_embedder()
        return self.vector_index.get_collection(model_name)
