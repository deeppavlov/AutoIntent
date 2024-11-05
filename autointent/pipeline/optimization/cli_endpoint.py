import logging

import hydra

from autointent import Context
from autointent.configs.optimization_cli import OptimizationConfig

from .pipeline_optimizer import PipelineOptimizer
from .utils import load_config


@hydra.main(config_name="optimization_config", config_path=".", version_base=None)
def main(cfg: OptimizationConfig) -> None:
    logger = logging.getLogger(__name__)

    logger.debug("Run Name: %s", cfg.logs.run_name)
    logger.debug("logs and assets: %s", cfg.logs.dirpath)
    logger.debug("Vector index path: %s", cfg.vector_index.db_dir)

    # create shared objects for a whole pipeline
    context = Context(cfg.seed)
    context.config_logs(cfg.logs)
    context.config_vector_index(cfg.vector_index, cfg.embedder)
    context.config_data(cfg.data, cfg.augmentation)

    # run optimization
    search_space_config = load_config(cfg.task.search_space_path, context.is_multilabel(), logger)
    pipeline = PipelineOptimizer.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    context.dump()
