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
        logs_path: str,
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
        self.optimization_logs = OptimizationLogs(logs_path)
        self.vector_index = VectorIndex(db_dir, device, self.data_handler.multilabel, self.data_handler.n_classes)

        self.device = device
        self.multilabel = self.data_handler.multilabel
        self.n_classes = self.data_handler.n_classes
        self.seed = seed

    def get_best_collection(self):
        model_name = self.optimization_logs.get_best_embedder()
        logger.info(f"Best embedder model name: {model_name}")
        if model_name is None:
            logger.warning("No best embedder found in optimization logs")
            return None
        return self.vector_index.get_collection(model_name)

    def print_all_fields(self):
        logger.info("Context fields:")
        logger.info(f"Device: {self.device}")
        logger.info(f"Multilabel: {self.multilabel}")
        logger.info(f"Number of classes: {self.n_classes}")
        logger.info(f"Seed: {self.seed}")
        self.data_handler.print_fields()
        self.vector_index.print_info()
