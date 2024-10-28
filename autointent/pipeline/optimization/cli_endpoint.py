import logging

import hydra

from autointent import Context
from autointent.configs.optimization_cli import OptimizationConfig

from .pipeline_optimizer import PipelineOptimizer
from .utils import load_config, load_data, setup_logging


@hydra.main(config_name="optimization_config", config_path=".", version_base=None)
def main(cfg: OptimizationConfig) -> None:
    setup_logging(cfg.logs.level.value)
    logger = logging.getLogger(__name__)

    logger.debug("Run Name: %s", cfg.logs.run_name)
    logger.debug("logs and assets: %s", cfg.logs.dirpath)
    logger.debug("Vector index path: %s", cfg.vector_index.db_dir)

    # create shared objects for a whole pipeline
    context = Context(
        load_data(cfg.data.train_path),
        None if cfg.data.test_path is None else load_data(cfg.data.test_path),
        cfg.vector_index.device,
        cfg.augmentation.multilabel_generation_config,
        cfg.augmentation.regex_sampling,
        cfg.seed,
        cfg.vector_index.db_dir,
        cfg.logs.dump_dir,
        cfg.data.force_multilabel,
        cfg.embedder.batch_size,
        cfg.embedder.max_length,
    )

    # run optimization
    search_space_config = load_config(cfg.task.search_space_path, context.multilabel, logger)
    pipeline = PipelineOptimizer.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    pipeline.dump(cfg.logs.dirpath)
