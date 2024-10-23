import logging

import hydra

from autointent import Context
from autointent.configs.optimization_cli import OptimizationConfig

from .pipeline_optimizer import PipelineOptimizer
from .utils import get_db_dir, get_logs_dir, get_run_name, load_config, load_data, setup_logging


@hydra.main(config_name="optimization_config", config_path=".", version_base=None)
def main(cfg: OptimizationConfig) -> None:
    setup_logging(cfg.logs.level.value)
    logger = logging.getLogger(__name__)

    # configure the run and data
    run_name = get_run_name(cfg.logs.run_name)
    db_dir = get_db_dir(run_name, cfg.vector_index.db_dir)
    logs_dir = get_logs_dir(run_name, cfg.logs.dirpath)
    dump_dir = logs_dir / "modules_dumps"

    logger.debug("Run Name: %s", run_name)
    logger.debug("Chroma DB path: %s", db_dir)

    # create shared objects for a whole pipeline
    context = Context(
        load_data(cfg.data.train_path),
        None if cfg.data.test_path is None else load_data(cfg.data.test_path),
        cfg.vector_index.device,
        cfg.augmentation.multilabel_generation_config,
        cfg.augmentation.regex_sampling,
        cfg.seed,
        db_dir,
        dump_dir,
        cfg.data.force_multilabel,
    )

    # run optimization
    search_space_config = load_config(cfg.task.search_space_path, context.multilabel, logger)
    pipeline = PipelineOptimizer.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir)
