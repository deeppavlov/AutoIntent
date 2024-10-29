import logging

import hydra

from autointent import Context
from autointent.configs.optimization_cli import OptimizationConfig

from .pipeline_optimizer import PipelineOptimizer
from .utils import get_db_dir, get_logs_dir, get_run_name, load_config, load_data


@hydra.main(config_name="optimization_config", config_path=".", version_base=None)
def main(cfg: OptimizationConfig) -> None:
    logger = logging.getLogger(__name__)

    # configure the run and data
    run_name = get_run_name(cfg.run_name)
    db_dir = get_db_dir(cfg.db_dir, run_name)
    logs_dir = get_logs_dir(cfg.logs_dir, run_name)
    dump_dir = str(logs_dir / "modules_dumps")

    logger.debug("Run Name: %s", run_name)
    logger.debug("Chroma DB path: %s", db_dir)

    # create shared objects for a whole pipeline
    context = Context(
        load_data(  # type: ignore[arg-type]
            cfg.dataset_path
        ),
        load_data(cfg.test_path),
        cfg.device,
        cfg.multilabel_generation_config,
        cfg.regex_sampling,
        cfg.seed,
        db_dir,
        dump_dir,
        cfg.force_multilabel,
        cfg.embedder_batch_size,
        cfg.embedder_max_length,
    )

    # run optimization
    search_space_config = load_config(cfg.search_space_path, context.multilabel, logger)
    pipeline = PipelineOptimizer.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir)
