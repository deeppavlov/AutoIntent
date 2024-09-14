import json
import os
from typing import List, Any
from pathlib import Path
from .pipeline.pipeline import Pipeline
from . import Context
from .pipeline.utils import get_db_dir, generate_name
from datetime import datetime

class AutoIntentAPI:
    def __init__(self, mode: str = "multiclass", device: str = "cuda:0"):
        self.mode = mode
        self.device = device
        self.pipeline = None
        self.context = None
        self.best_pipeline_path = Path("best_pipeline.json")

    def fit(self, multiclass_data: List[Any], multilabel_data: List[Any], test_data: List[Any], hyperparameters: dict):
        config_path = hyperparameters.get('config_path', '')
        run_name = hyperparameters.get('run_name', generate_name())
        run_name = f"{run_name}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
        db_dir = get_db_dir(hyperparameters.get('db_dir', ''), run_name)

        # Проверяем, существует ли сохраненный лучший пайплайн
        if os.path.exists(self.best_pipeline_path):
            with open(self.best_pipeline_path, 'r') as f:
                saved_pipeline = json.load(f)

            # Проверяем, изменились ли гиперпараметры
            if saved_pipeline['hyperparameters'] == hyperparameters:
                print("Loading saved pipeline...")
                self.pipeline = Pipeline.load(saved_pipeline['pipeline'])
                self.pipeline.best_modules = saved_pipeline['best_modules']
                return

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

        self.pipeline = Pipeline(config_path, self.mode, verbose=hyperparameters.get('verbose', False))
        self.pipeline.optimize(self.context)

        # Сохранение лучшего пайплайна
        best_pipeline = {
            'pipeline': self.pipeline.serialize(),
            'hyperparameters': hyperparameters,
            'best_modules': self.pipeline.best_modules
        }
        with open(self.best_pipeline_path, 'w') as f:
            json.dump(best_pipeline, f)

        # Сохранение результатов в логи
        logs_dir = hyperparameters.get('logs_dir', '')
        if logs_dir:
            self.pipeline.dump(logs_dir, run_name)

    def predict(self, texts: List[str]) -> List[Any]:
        if self.pipeline is None:
            raise ValueError("Pipeline is not fitted. Call fit() first.")
        return self.pipeline.predict(texts)