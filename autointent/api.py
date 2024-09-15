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
        self.best_pipeline_path = Path("best_pipeline.json")

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
            hyperparameters.get('seed', 0)
        )

        # Проверяем, существует ли сохраненный лучший пайплайн
        if os.path.exists(self.best_pipeline_path):
            with open(self.best_pipeline_path, 'r') as f:
                saved_pipeline = json.load(f)

            # Проверяем, изменились ли гиперпараметры
            if saved_pipeline['hyperparameters'] == hyperparameters:
                print("Loading saved pipeline...")
                self.pipeline = Pipeline(config_path, self.mode,
                                         verbose=hyperparameters.get('verbose', False))
                self.pipeline.load_best_modules(saved_pipeline['best_modules'], self.context)
                return

        self.pipeline = Pipeline(config_path, self.mode,
                                 verbose=hyperparameters.get('verbose', False))
        self.pipeline.optimize(self.context)

        # Сохранение лучшего пайплайна
        best_pipeline = {
            'hyperparameters': hyperparameters,
            'best_modules': self.pipeline.save_best_modules()
        }
        with open(self.best_pipeline_path, 'w') as f:
            json.dump(best_pipeline, f)

        # Сохранение результатов в логи
        logs_dir = hyperparameters.get('logs_dir', '')
        if logs_dir:
            self.pipeline.dump(logs_dir, run_name)

    def predict(self, texts: List[str], intents_dict) -> List[Any]:
        if self.pipeline is None:
            raise ValueError("Pipeline is not fitted. Call fit() first.")
        return self.pipeline.predict(texts, intents_dict)

    def _save_best_modules(self, best_modules):
        saved_modules = {}
        for node_type, module_config in best_modules.items():
            saved_modules[node_type] = {
                'module_type': module_config['module_type'],
                'parameters': {k: v for k, v in module_config.items() if k != 'module_type'}
            }
        return saved_modules

    def _load_best_modules(self, saved_modules):
        loaded_modules = {}
        for node_type, module_info in saved_modules.items():
            node_class = self.pipeline.available_nodes[node_type]
            module_class = node_class.modules_available[module_info['module_type']]
            # Создаем экземпляр модуля
            loaded_modules[node_type] = module_class(**module_info['parameters'])
        return loaded_modules
