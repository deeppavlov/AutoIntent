from .data_handler import DataHandler
from .optimization_logs import OptimizationLogs
from .vector_index import VectorIndex
import logging

logger = logging.getLogger(__name__)

class Context:
    def __init__(
        self,
        multiclass_intent_records,
        multilabel_utterance_records,
        test_utterance_records,
        device,
        mode,
        multilabel_generation_config: str,
        db_dir,
        regex_sampling,
        seed,
    ) -> None:
        self.data_handler = DataHandler(
            multiclass_intent_records,
            multilabel_utterance_records,
            test_utterance_records,
            mode,
            multilabel_generation_config,
            regex_sampling,
            seed,
        )
        self.optimization_logs = OptimizationLogs()
        self.vector_index = VectorIndex(db_dir, device, self.data_handler.multilabel, self.data_handler.n_classes)

        self.device = device
        self.multilabel = self.data_handler.multilabel
        self.n_classes = self.data_handler.n_classes
        self.seed = seed

    def print_all_fields(self):
        logger.info("Context fields:")
        logger.info(f"Device: {self.device}")
        logger.info(f"Multilabel: {self.multilabel}")
        logger.info(f"Number of classes: {self.n_classes}")
        logger.info(f"Seed: {self.seed}")
        logger.info("Data Handler fields:")
        self.data_handler.print_fields()
        logger.info("Optimization Logs:")
        self.optimization_logs.print_logs()
        logger.info("Vector Index:")
        self.vector_index.print_info()


    def get_best_collection(self):
        model_name = self.optimization_logs.get_best_embedder()
        print(model_name)
        return self.vector_index.get_collection(model_name)
