"""CLI for evolutionary augmenter."""

import logging
from argparse import ArgumentParser, Namespace

from autointent import load_dataset
from autointent.generation.utterances.evolution import IncrementalUtteranceEvolver, UtteranceEvolver
from autointent.generation.utterances.generator import Generator

from .chat_templates import (
    AbstractEvolution,
    ConcreteEvolution,
    FormalEvolution,
    FunnyEvolution,
    GoofyEvolution,
    InformalEvolution,
    ReasoningEvolution,
)

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


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
    evolutions = []

    for arg_name, evolution_cls in mapping.items():
        if getattr(args, arg_name):
            evolutions.append(evolution_cls())  # type: ignore[abstract]

    if not evolutions:
        logger.warning("No evolutions selected. Exiting.")
        return

    utterance_evolver: UtteranceEvolver
    if args.decide_for_me:
        utterance_evolver = IncrementalUtteranceEvolver(Generator(), evolutions, args.seed, args.async_mode)
    else:
        utterance_evolver = UtteranceEvolver(Generator(), evolutions, args.seed, args.async_mode)
    dataset = load_dataset(args.input_path)

    n_before = len(dataset[args.split])

    new_samples = utterance_evolver.augment(
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
