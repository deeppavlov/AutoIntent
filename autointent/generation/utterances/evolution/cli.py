"""CLI for evolutionary augmenter."""

import logging
from argparse import ArgumentParser

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
            {"module_name": "argmax"},
        ],
    },
]


def optimize_n_evolutions(
    input_path: str, max_n_evolutions: int, evolutions: list, seed: int, split: str
) -> tuple[Dataset, int]:
    emb_config = EmbedderConfig(batch_size=16, device="cuda")

    best_result = 0
    best_n = 0
    for n in range(max_n_evolutions):
        dataset = load_dataset(input_path)
        generator = UtteranceEvolver(Generator(), evolutions, seed)
        generator.augment(dataset, split_name=split, n_evolutions=n)

        pipeline_optimizer = Pipeline.from_search_space(SEARCH_SPACE)
        pipeline_optimizer.set_config(emb_config)
        ctx = pipeline_optimizer.fit(dataset)
        results = ctx.optimization_info.dump_evaluation_results()
        decision_metric = results["metrics"]["decision"]

        if decision_metric > best_result:
            best_result = decision_metric
            best_n = n
        else:
            break
    return dataset, best_n


def main() -> None:
    """CLI endpoint."""
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
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    evolutions: list[EvolutionChatTemplate] = []
    if args.reasoning:
        evolutions.append(ReasoningEvolution())
    if args.concretizing:
        evolutions.append(ConcreteEvolution())
    if args.abstract:
        evolutions.append(AbstractEvolution())
    if args.formal:
        evolutions.append(FormalEvolution())
    if args.funny:
        evolutions.append(FunnyEvolution())
    if args.goofy:
        evolutions.append(GoofyEvolution())
    if args.informal:
        evolutions.append(InformalEvolution())

    if not evolutions:
        logger.warning("No evolutions selected. Exiting.")
        return

    if args.decide_for_me:
        dataset, n = optimize_n_evolutions(args.input_path, args.n_evolutions, evolutions, args.seed, args.split)
        logger.info("# optimal n evolutions: %s", n)
    else:
        dataset = load_dataset(args.input_path)
        n_before = len(dataset[args.split])

        generator = UtteranceEvolver(Generator(), evolutions, args.seed)
        new_samples = generator.augment(dataset, split_name=args.split, n_evolutions=args.n_evolutions)
        n_after = len(dataset[args.split])

        logger.info("# samples before %s", n_before)
        logger.info("# samples generated %s", len(new_samples))
        logger.info("# samples after %s", n_after)

    dataset.to_json(args.output_path)

    if args.output_repo is not None:
        dataset.push_to_hub(args.output_repo)


if __name__ == "__main__":
    main()
