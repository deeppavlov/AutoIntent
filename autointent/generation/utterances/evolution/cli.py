"""CLI for evolutionary augmenter."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from datasets import concatenate_datasets

from autointent import Dataset, Pipeline, load_dataset
from autointent.configs import EmbedderConfig
from autointent.generation.utterances.evolution.evolver import UtteranceEvolver
from autointent.generation.utterances.generator import Generator

from .chat_templates import (
    AbstractEvolution,
    ConcreteEvolution,
    EvolutionChatTemplate,
    FormalEvolution,
    FunnyEvolution,
    GoofyEvolution,
    InformalEvolution,
    ReasoningEvolution,
)

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

SEARCH_SPACE = [
    {
        "node_type": "embedding",
        "target_metric": "retrieval_hit_rate",
        "search_space": [
            {
                "module_name": "retrieval",
                "k": [5],
                "embedder_name": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                ],
            }
        ],
    },
    {
        "node_type": "scoring",
        "target_metric": "scoring_roc_auc",
        "metrics": ["scoring_accuracy"],
        "search_space": [{"module_name": "linear"}],
    },
    {
        "node_type": "decision",
        "target_metric": "decision_accuracy",
        "search_space": [
            {"module_name": "tunable"},
        ],
    },
]


def _choose_search_space(search_space: str | None) -> list[dict[str, Any]] | Path | str:
    if search_space is None:
        return SEARCH_SPACE
    return search_space


def _optimize_n_evolutions(
    generator: Generator,
    input_path: str,
    dataset: Dataset,
    max_n_evolutions: int,
    split_train: str,
    batch_size: int,
    search_space: str | None,
) -> Dataset:
    emb_config = EmbedderConfig(batch_size=16, device="cuda")
    search_space = _choose_search_space(search_space)

    best_result = 0
    best_n = 0
    merge_dataset = load_dataset(input_path)

    for n in range(max_n_evolutions):
        new_samples_dataset = generator.augment(
            dataset, split_name=split_train, n_evolutions=1, update_split=False, batch_size=batch_size
        )
        merge_dataset[split_train] = concatenate_datasets([merge_dataset[split_train], new_samples_dataset])

        pipeline_optimizer = Pipeline.from_search_space(search_space)
        pipeline_optimizer.set_config(emb_config)
        ctx = pipeline_optimizer.fit(merge_dataset)
        results = ctx.optimization_info.dump_evaluation_results()
        decision_metric = results["metrics"]["decision"][0]

        if decision_metric > best_result:
            best_result = decision_metric
            best_n = n
        else:
            break

    logger.info("# optimal n evolutions: %s", best_n)
    return dataset


def _parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to json or hugging face repo with dataset",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Local path where to save result",
    )
    parser.add_argument(
        "--output-repo",
        type=str,
        default=None,
        help="Local path where to save result",
    )
    parser.add_argument("--private", action="store_true", help="Publish privately if --output-repo option is used")
    parser.add_argument("--n-evolutions", type=int, default=1, help="Number of utterances to generate for each intent")
    parser.add_argument("--decide-for-me", action="store_true")
    parser.add_argument("--reasoning", action="store_true", help="Whether to use `Reasoning` evolution")
    parser.add_argument("--concretizing", action="store_true", help="Whether to use `Concretizing` evolution")
    parser.add_argument("--abstract", action="store_true", help="Whether to use `Abstract` evolution")
    parser.add_argument("--formal", action="store_true", help="Whether to use `Formal` evolution")
    parser.add_argument("--funny", action="store_true", help="Whether to use `Funny` evolution")
    parser.add_argument("--goofy", action="store_true", help="Whether to use `Goofy` evolution")
    parser.add_argument("--informal", action="store_true", help="Whether to use `Informal` evolution")
    parser.add_argument("--async-mode", action="store_true", help="Enable asynchronous generation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--search-space", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    """CLI endpoint."""
    mapping = {
        "reasoning": ReasoningEvolution,
        "concretizing": ConcreteEvolution,
        "abstract": AbstractEvolution,
        "formal": FormalEvolution,
        "funny": FunnyEvolution,
        "goofy": GoofyEvolution,
        "informal": InformalEvolution,
    }
    args = _parse_args()
    evolutions: list[EvolutionChatTemplate] = []

    for arg_name, evolution_cls in mapping.items():
        if getattr(args, arg_name):
            evolutions.append(evolution_cls())

    if not evolutions:
        logger.warning("No evolutions selected. Exiting.")
        return

    generator = UtteranceEvolver(Generator(), evolutions, args.seed, args.async_mode)
    dataset = load_dataset(args.input_path)

    if args.decide_for_me:
        dataset = _optimize_n_evolutions(
            generator,
            args.input_path,
            dataset,
            args.n_evolutions,
            args.split,
            args.batch_size,
            args.search_space,
        )
    else:
        n_before = len(dataset[args.split])

        new_samples = generator.augment(
            dataset, split_name=args.split, n_evolutions=args.n_evolutions, batch_size=args.batch_size
        )
        n_after = len(dataset[args.split])

        logger.info("# samples before %s", n_before)
        logger.info("# samples generated %s", len(new_samples))
        logger.info("# samples after %s", n_after)

    dataset.to_json(args.output_path)

    if args.output_repo is not None:
        dataset.push_to_hub(args.output_repo)


if __name__ == "__main__":
    main()
