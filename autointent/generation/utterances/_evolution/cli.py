"""CLI for evolutionary augmenter."""

import logging
from argparse import ArgumentParser, Namespace

from autointent import load_dataset
from autointent.generation import Generator
from autointent.generation.chat_templates import (
    EVOLUTION_MAPPING,
    EVOLUTION_NAMES,
)

from .evolver import UtteranceEvolver
from .incremental_evolver import IncrementalUtteranceEvolver

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
    parser.add_argument("--decide-for-me", action="store_true", help="Enable incremental evolution")
    parser.add_argument("--template", type=str, choices=EVOLUTION_NAMES, help="Template to use", nargs="+")
    parser.add_argument("--async-mode", action="store_true", help="Enable asynchronous generation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--search-space", type=str, default=None)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Use sequential evolution. When this option is enabled, solutions "
            "will evolve one after another, instead of using a parallel approach."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """CLI endpoint."""
    args = _parse_args()
    evolutions = [EVOLUTION_MAPPING[template_name] for template_name in args.template]

    utterance_evolver: UtteranceEvolver
    if args.decide_for_me:
        utterance_evolver = IncrementalUtteranceEvolver(Generator(), evolutions, args.seed, args.async_mode)
    else:
        utterance_evolver = UtteranceEvolver(Generator(), evolutions, args.seed, args.async_mode)
    dataset = load_dataset(args.input_path)

    n_before = len(dataset[args.split])

    new_samples = utterance_evolver.augment(
        dataset,
        split_name=args.split,
        n_evolutions=args.n_evolutions,
        batch_size=args.batch_size,
        sequential=args.sequential,
    )
    n_after = len(dataset[args.split])

    logger.info("# samples before %s", n_before)
    logger.info("# samples generated %s", len(new_samples))
    logger.info("# samples after %s", n_after)

    dataset.to_json(args.output_path)

    if args.output_repo is not None:
        dataset.push_to_hub(args.output_repo, args.private)
