import logging

import hydra

from autointent import Context
from autointent.configs.optimization_cli import OptimizationConfig

from .pipeline_optimizer import PipelineOptimizer
from .utils import get_db_dir, get_logs_dir, get_run_name, load_config, load_data, setup_logging

# def main():
#     parser = ArgumentParser()
#     parser.add_argument(
#         "--config-path",
#         type=str,
#         default="",
#         help="Path to a yaml configuration file that defines the optimization search space. "
#         "Omit this to use the default configuration.",
#     )
#     parser.add_argument(
#         "--multiclass-path",
#         type=str,
#         default="",
#         help="Path to a json file with intent records. "
#         'Set to "default" to use banking77 data stored within the autointent package.',
#     )
#     parser.add_argument(
#         "--multilabel-path",
#         type=str,
#         default="",
#         help="Path to a json file with utterance records. "
#         'Set to "default" to use dstc3 data stored within the autointent package.',
#     )
#     parser.add_argument(
#         "--test-path",
#         type=str,
#         default="",
#         help="Path to a json file with utterance records. "
#         "Skip this option if you want to use a random subset of the training sample as test data.",
#     )
#     parser.add_argument(
#         "--db-dir",
#         type=str,
#         default="",
#         help="Location where to save chroma database file. Omit to use your system's default cache directory.",
#     )
#     parser.add_argument(
#         "--logs-dir",
#         type=str,
#         default="",
#         help="Location where to save optimization logs that will be saved as "
#         "`<logs_dir>/<run_name>_<cur_datetime>/logs.json`",
#     )
#     parser.add_argument(
#         "--run-name", type=str, default="", help="Name of the run prepended to optimization logs filename"
#     )
#     parser.add_argument(
#         "--mode",
#         choices=["multiclass", "multilabel", "multiclass_as_multilabel"],
#         default="multiclass",
#         help="Evaluation mode. This parameter must be consistent with provided data.",
#     )
#     parser.add_argument("--device", type=str, default="cuda:0", help="Specify device in torch notation")
#     parser.add_argument(
#         "--regex-sampling",
#         type=int,
#         default=0,
#         help="Number of shots per intent to sample from regular expressions. "
#         "This option extends sample utterances within multiclass intent records.",
#     )
#     parser.add_argument("--seed", type=int, default=0, help="Affects the data partitioning")
#     parser.add_argument(
#         "--log-level", type=str, default="ERROR", choices=LoggingLevelType.__args__, help="Set the logging level"
#     )
#     parser.add_argument(
#         "--multilabel-generation-config",
#         type=str,
#         default="",
#         help='Config string like "[20, 40, 20, 10]" means 20 one-label examples, '
#         "40 two-label examples, 20 three-label examples, 10 four-label examples. "
#         "This option extends multilabel utterance records.",
#     )
#     args = parser.parse_args()

#     setup_logging(args.log_level)
#     logger = logging.getLogger(__name__)

#     # configure the run and data
#     run_name = get_run_name(args.run_name)
#     db_dir = get_db_dir(args.db_dir, run_name)

#     logger.debug("Run Name: %s", run_name)
#     logger.debug("Chroma DB path: %s", db_dir)

#     # create shared objects for a whole pipeline
#     context = Context(
#         load_data(args.multiclass_path, multilabel=False),
#         load_data(args.multilabel_path, multilabel=True),
#         load_data(args.test_path, multilabel=True),
#         args.device,
#         args.mode,
#         args.multilabel_generation_config,
#         db_dir,
#         args.regex_sampling,
#         args.seed,
#     )

#     # run optimization
#     search_space_config = load_config(args.config_path, context.multilabel, logger)
#     pipeline: Pipeline = instantiate(PipelineOptimizerConfig(), **search_space_config)
#     pipeline.optimize(context)

#     # save results
#     pipeline.dump(args.logs_dir, run_name)


@hydra.main(config_name="optimization_config", config_path=".", version_base=None)
def main(cfg: OptimizationConfig) -> None:
    setup_logging(cfg.log_level.value)
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
        load_data(cfg.dataset_path, multilabel=False),
        load_data(cfg.test_path, multilabel=True),
        cfg.device,
        cfg.multilabel_generation_config,
        cfg.regex_sampling,
        cfg.seed,
        db_dir,
        dump_dir,
    )

    # run optimization
    search_space_config = load_config(cfg.search_space_path, context.multilabel, logger)
    pipeline = PipelineOptimizer.from_dict_config(search_space_config)
    pipeline.optimize(context)

    # save results
    pipeline.dump(logs_dir)
