"""CLI for evolutionary augmenter."""

import logging
from argparse import ArgumentParser

from autointent import load_dataset
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
    parser.add_argument("--reasoning", action="store_true", help="Whether to use `Reasoning` evolution")
    parser.add_argument("--concretizing", action="store_true", help="Whether to use `Concretizing` evolution")
    parser.add_argument("--abstract", action="store_true", help="Whether to use `Abstract` evolution")
    parser.add_argument("--formal", action="store_true", help="Whether to use `Formal` evolution")
    parser.add_argument("--funny", action="store_true", help="Whether to use `Funny` evolution")
    parser.add_argument("--goofy", action="store_true", help="Whether to use `Goofy` evolution")
    parser.add_argument("--informal", action="store_true", help="Whether to use `Informal` evolution")
    parser.add_argument("--async-mode", action="store_true", help="Enable asynchronous generation")
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

    dataset = load_dataset(args.input_path)
    n_before = len(dataset[args.split])

    generator = UtteranceEvolver(Generator(), evolutions, args.seed, async_mode=args.async_mode)
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
