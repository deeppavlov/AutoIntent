import json
import os
import logging

from typing import List, Any
from pathlib import Path
from .pipeline.pipeline import Pipeline
from . import Context
from .pipeline.utils import get_db_dir, generate_name
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoIntentAPI:
    def __init__(self, mode: str = "multiclass", device: str = "cuda:0"):
        self.mode = mode
        self.device = device
        self.pipeline = None
        self.context = None
        self.logs_path = Path("optimization_logs.json")

    def fit(self, multiclass_data: List[Any], multilabel_data: List[Any], test_data: List[Any],
            hyperparameters: dict):
        config_path = hyperparameters.get('config_path', '')
        run_name = hyperparameters.get('run_name', generate_name())
        run_name = f"{run_name}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
        db_dir = get_db_dir(hyperparameters.get('db_dir', ''), run_name)

        self.context = Context(
            multiclass_data,
            multilabel_data,
            test_data,
            self.device,
            self.mode,
            hyperparameters.get('multilabel_generation_config', ''),
            db_dir,
            hyperparameters.get('regex_sampling', 0),
            hyperparameters.get('seed', 0),
            self.logs_path
        )
        logger.debug("Checking if logs_path exists")
        if os.path.exists(self.logs_path):
            logger.debug("logs_path exists")
            logger.debug("Found saved optimization logs. Attempting to load...")
            self.pipeline = Pipeline(config_path, self.mode,
                                     verbose=hyperparameters.get('verbose', False))
            try:
                best_modules = self.context.optimization_logs.get_best_modules()
                self.pipeline.best_modules = self._load_best_modules(best_modules)
                logger.debug("Successfully loaded saved pipeline.")
                return
            except Exception as e:
                logger.warning(f"Failed to load saved pipeline: {str(e)}. Will re-optimize.")

        self.pipeline = Pipeline(config_path, self.mode,
                                 verbose=hyperparameters.get('verbose', False))
        self.pipeline.optimize(self.context)

        logs_dir = hyperparameters.get('logs_dir', '')
        if logs_dir:
            self.pipeline.dump(logs_dir, run_name)

    def _load_best_modules(self, best_modules):
        loaded_modules = {}
        for node_type, module_info in best_modules.items():
            node_class = self.pipeline.available_nodes[node_type]
            module_class = node_class.modules_available[module_info['module_type']]
            valid_params = {k: v for k, v in module_info.items()
                            if
                            k in module_class.__init__.__code__.co_varnames and k != 'module_type'}
            module = module_class(**valid_params)

            if hasattr(module, 'fit'):
                module.fit(self.context)

            loaded_modules[node_type] = module
        return loaded_modules

    def predict(self, texts: List[str], intents_dict) -> List[Any]:
        if self.pipeline is None:
            raise ValueError("Pipeline is not fitted. Call fit() first.")
        return self.pipeline.predict(texts, intents_dict)

    def _save_best_modules(self, best_modules):
        saved_modules = {}
        for node_type, module_config in best_modules.items():
            module_class = type(module_config)
            valid_params = {k: v for k, v in module_config.__dict__.items()
                            if k in module_class.__init__.__code__.co_varnames}
            saved_modules[node_type] = {
                'module_type': module_config.__class__.__name__,
                'parameters': valid_params
            }
        return saved_modules
